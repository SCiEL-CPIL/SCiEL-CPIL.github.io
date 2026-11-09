import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Evaluator limit is 120s. We use 110s to be safe for Python overhead.
    deadline = start_time + 110.0

    SQRT3 = math.sqrt(3)

    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            # Normals are perpendicular to edges. Edges are at 30, 90... relative to center?
            # For a regular hexagon with vertices at 0, 60, 120...
            # Edge 0 connects v0 and v1. Midpoint angle is 30. Normal is 30+90=120?
            # Let's stick to the previous working definition:
            # Vertices at i*60. Edge between i and i+1.
            # Normal to edge i is at i*60 + 30 + 90? No.
            # Vector v_i to v_{i+1} is rotated by 60.
            # The normal to the edge connecting vertex i and i+1 is at angle (i*60 + 30) + 90?
            # Actually, for a hexagon aligned with flat top/bottom (vertex at 0), 
            # the apothem directions are 30, 90, 150, 210, 270, 330.
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, d_sq):
        """
        Fast disjoint check.
        1. If d_sq > 4.0 (d > 2.0), definitely disjoint (min dist for touching is 2? No, 
           min center distance for disjoint unit hexagons is sqrt(3) if aligned flat-to-flat?
           Actually, if they touch at a vertex, dist can be smaller?
           Min distance between centers of two disjoint unit hexagons:
           If they touch edge-to-edge, dist is sqrt(3) * 2? No.
           Apothem is sqrt(3)/2. Distance between parallel edges is sqrt(3).
           So if centers are closer than sqrt(3) along the normal, they overlap.
           But if rotated, they might be further apart.
           Safe bound: if d_sq > (2 * (sqrt(3)/2) * 2)^2? No.
           Max distance between any two points in hex is 2.
           If d > 4, definitely disjoint.
           If d < sqrt(3) approx 1.732, they *might* overlap, but not always if rotated.
           However, if d < sqrt(3), they *could* be disjoint if rotated 30 deg?
           Let's use a tighter bound for "definitely overlap" to save SAT time.
           If d_sq < 1.0, they definitely overlap (center is inside other).
           If d_sq < 3.0, it's a "maybe".
           Let's use d_sq < 2.5 as a threshold for "likely overlap" to force SAT?
           Actually, the previous code used d_sq < 3.0 -> False (overlap).
           Is it possible for two unit hexagons to be disjoint with center distance < sqrt(3)?
           Yes, if rotated 30 degrees relative to each other, the "gap" at the vertex allows 
           centers to be closer? No, the inradius is sqrt(3)/2.
           If they touch at a vertex, the distance is 2 * sqrt(3)/2 = sqrt(3).
           So if d < sqrt(3), they MUST overlap.
           So d_sq < 3.0 -> Overlap is correct.
        """
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
            
        # SAT check
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        # The outer hexagon contains the points if the projection of every point
        # onto the 6 normal axes is within [-S*sqrt(3)/2, S*sqrt(3)/2].
        # So S >= 2 * max_proj / sqrt(3).
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial configurations
    # 1. Flower + extensions (compact)
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # 2. Compact 3-5-3
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # 3. Linear
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    configs = [coords1, coords2, coords3]
    
    best_params = None
    best_score = float('inf')
    
    # STRATEGY:
    # Phase 1: Rigid Cluster Optimization
    # Treat the cluster as a rigid body with a "scale" parameter.
    # We optimize: Translation (x,y), Rotation (theta), Scale (s), Outer Angle (oa).
    # The relative positions are fixed from the initial config.
    # This reduces search space to 5 dimensions per config.
    
    def evaluate_rigid(base_centers, base_angles, tx, ty, theta_deg, scale, outer_angle_deg):
        """
        Evaluates a rigid transformation of the base cluster.
        base_centers: list of (x,y)
        base_angles: list of angles
        tx, ty: translation
        theta_deg: global rotation
        scale: uniform scaling of the cluster
        outer_angle_deg: rotation of the outer hexagon
        """
        # Apply rigid transform
        centers = []
        angles = []
        cos_t = math.cos(math.radians(theta_deg))
        sin_t = math.sin(math.radians(theta_deg))
        
        for i in range(N):
            bx, by = base_centers[i]
            ba = base_angles[i]
            
            # Scale
            sx = bx * scale
            sy = by * scale
            
            # Rotate
            rx = sx * cos_t - sy * sin_t
            ry = sx * sin_t + sy * cos_t
            
            # Translate
            cx = rx + tx
            cy = ry + ty
            
            # Angle: global rotation + base angle
            # Note: Scaling doesn't change angle, rotation adds to angle.
            a = ba + theta_deg
            
            centers.append((cx, cy))
            angles.append(a)
            
        # Center the cluster at origin for outer hex calculation?
        # The problem allows the outer hex to be anywhere, but we can WLOG center the cluster
        # at (0,0) and optimize the outer hex position/angle?
        # The interface asks for: inner (x,y,angle), outer (x,y), side, angle.
        # If we center the cluster at (0,0), the outer hex is best centered at (0,0) too.
        # So we just need to find the min side length for a hexagon centered at (0,0).
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    overlap_penalty += 1000.0
        
        side = calc_outer_side(hexagons, outer_angle_deg)
        return overlap_penalty + side

    # Phase 1: Optimize Rigid Parameters
    for config in configs:
        if time.time() > deadline - 40.0:
            break
            
        base_centers = []
        base_angles = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            base_centers.append((x, y))
            base_angles.append(0.0)
            
        # Initial guess for rigid params
        # tx, ty = 0, 0
        # theta = 0
        # scale = 1.0
        # outer_angle = 0
        
        # Use Nelder-Mead for the 5-dimensional rigid search
        # Variables: tx, ty, theta_deg, scale, outer_angle_deg
        def objective_rigid(p):
            tx, ty, theta_deg, scale, outer_angle_deg = p
            return evaluate_rigid(base_centers, base_angles, tx, ty, theta_deg, scale, outer_angle_deg)
            
        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        
        try:
            res = minimize(objective_rigid, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                # Store the config index to know which base to use for Phase 2
                best_config_idx = configs.index(config)
        except Exception:
            pass

    # Phase 2: Local Micro-Optimization (Breaking Rigidity)
    # Start from the best rigid configuration.
    # Now allow individual hexagons to move slightly.
    # This is a standard SA or local search, but initialized very close to the global optimum.
    
    if best_params is not None:
        tx, ty, theta_deg, scale, outer_angle_deg = best_params
        config_idx = best_config_idx if 'best_config_idx' in locals() else 0
        base_config = configs[config_idx]
        
        # Generate the specific centers and angles from the rigid params
        base_centers = []
        base_angles = []
        for q, r in base_config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            base_centers.append((x, y))
            base_angles.append(0.0)
            
        cos_t = math.cos(math.radians(theta_deg))
        sin_t = math.sin(math.radians(theta_deg))
        
        current_centers = []
        current_angles = []
        for i in range(N):
            bx, by = base_centers[i]
            ba = base_angles[i]
            
            sx = bx * scale
            sy = by * scale
            
            rx = sx * cos_t - sy * sin_t
            ry = sx * sin_t + sy * cos_t
            
            cx = rx + tx
            cy = ry + ty
            a = ba + theta_deg
            
            current_centers.append((cx, cy))
            current_angles.append(a)
            
        # Now run a short SA on the 33-dimensional space (x,y,angle for each)
        # but with very small steps, starting from this good point.
        
        def evaluate_full(centers, angles, outer_angle_deg):
            hexagons = []
            hex_axes = []
            for i in range(N):
                cx, cy = centers[i]
                a = angles[i]
                hex_verts = get_hex_vertices(cx, cy, a)
                hexagons.append(hex_verts)
                hex_axes.append(get_hex_axes(a))
            
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d_sq = dx*dx + dy*dy
                    if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                        overlap_penalty += 1000.0
            
            side = calc_outer_side(hexagons, outer_angle_deg)
            return overlap_penalty + side

        current_score = evaluate_full(current_centers, current_angles, outer_angle_deg)
        local_best_score = current_score
        local_best_centers = [c.copy() for c in current_centers]
        local_best_angles = current_angles.copy()
        local_best_oa = outer_angle_deg
        
        T = 0.5
        T_min = 0.01
        cooling_rate = 0.99
        steps = 0
        max_steps = 500
        
        while T > T_min and steps < max_steps and time.time() < deadline - 10.0:
            # Perturb one hexagon
            idx = random.randint(0, N-1)
            new_centers = [c.copy() for c in current_centers]
            new_angles = current_angles.copy()
            
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            da = random.gauss(0, 5.0)
            
            new_centers[idx] = (new_centers[idx][0] + dx, new_centers[idx][1] + dy)
            new_angles[idx] += da
            
            # Also slightly perturb outer angle
            new_oa = outer_angle_deg + random.gauss(0, 0.5)
            
            new_score = evaluate_full(new_centers, new_angles, new_oa)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_centers = new_centers
                current_angles = new_angles
                outer_angle_deg = new_oa
                current_score = new_score
                
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_centers = [c.copy() for c in current_centers]
                    local_best_angles = current_angles.copy()
                    local_best_oa = outer_angle_deg
            
            steps += 1
            if steps % 10 == 0:
                T *= cooling_rate
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = None # We will use local_best directly
            final_centers = local_best_centers
            final_angles = local_best_angles
            final_oa = local_best_oa
        else:
            # Use the rigid result if it was better
            final_centers = current_centers
            final_angles = current_angles
            final_oa = outer_angle_deg
            best_score = current_score # Update to be sure

    else:
        # Fallback if Phase 1 failed
        base_config = configs[0]
        final_centers = []
        final_angles = []
        for q, r in base_config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            final_centers.append((x, y))
            final_angles.append(0.0)
        final_oa = 0.0
        best_score = evaluate_full(final_centers, final_angles, final_oa)

    # Final Validation and Return
    # Center the final configuration at (0,0) for the output?
    # The outer hexagon is at (0,0).
    # We should shift the inner hexagons so their centroid is at (0,0) to minimize outer side?
    # Actually, calc_outer_side assumes outer hex is at (0,0).
    # If we shift the inner hexagons, the max_proj changes.
    # The optimal outer hexagon for a set of points is centered at the centroid of the points?
    # Not exactly, but close.
    # Let's shift the final_centers so their average is (0,0).
    
    cx_avg = sum(c[0] for c in final_centers) / N
    cy_avg = sum(c[1] for c in final_centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in final_centers]
    
    # Recalculate side with shifted centers
    hexagons = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = final_angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
    
    # Optimize outer angle one last time with the shifted centers
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    # Final overlap check
    valid = True
    hex_axes = [get_hex_axes(final_angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            dx = shifted_centers[i][0] - shifted_centers[j][0]
            dy = shifted_centers[i][1] - shifted_centers[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to the initial config 1, centered
        inner_list = []
        for q, r in configs[0]:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side_fb = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side_fb, 0.0)

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], final_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
