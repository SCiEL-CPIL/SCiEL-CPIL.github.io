import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (evaluator limit 150s)
    
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
        1. If d_sq > 4.0 (d > 2.0), definitely disjoint.
        2. If d_sq < 3.0 (d < sqrt(3)), definitely overlap (min distance for disjoint is sqrt(3)).
        3. Otherwise, use SAT.
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
        
        # Apothem of outer hex is S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # --- SMOOTH OBJECTIVE FOR SLSQP ---
    
    def smooth_max(values, beta):
        """
        Smooth approximation of max(values).
        As beta -> inf, this approaches the true max.
        """
        if len(values) == 0: return 0.0
        max_v = max(values)
        # To avoid overflow in exp, subtract max_v
        exp_sum = 0.0
        for v in values:
            exp_sum += math.exp(beta * (v - max_v))
        return max_v + math.log(exp_sum) / beta

    def signed_distance_hexagons(c1, c2, a1, a2):
        """
        Approximate signed distance between two unit hexagons.
        Positive if disjoint, negative if overlapping.
        We use the distance between centers minus the sum of 'effective radii' along the connecting line,
        but for hexagons, a better metric for smoothness in optimization is based on the 
        maximum projection overlap.
        
        However, calculating exact SAT distance is non-differentiable.
        We approximate the constraint: d(c1, c2) >= d_min(a1, a2).
        d_min depends on relative orientation.
        
        A robust smooth proxy:
        Let d = distance between centers.
        Let r_eff = max projection radius of hex1 + max projection radius of hex2 along the vector connecting centers?
        No, that's not quite right for arbitrary orientations.
        
        Let's use the distance between centers minus the sum of the apothems (1.5) as a lower bound?
        Min distance for disjointness is sqrt(3) ~ 1.732.
        
        Better approach: Use the "smooth SAT" concept.
        For each of the 12 axes, calculate the gap.
        Gap_i = (max_proj1_i - min_proj2_i) OR (max_proj2_i - min_proj1_i).
        If Gap_i > 0, they are separated by that amount on that axis.
        If all Gap_i > 0, they are disjoint.
        The "penetration depth" is roughly -min(Gap_i) if min < 0.
        
        We want to minimize the max penetration.
        Penetration = max(0, -min(Gap_i)).
        Smooth Penetration = smooth_max(0, -min(Gap_i)).
        -min(Gap_i) = max(-Gap_i).
        So Smooth Penetration = smooth_max(0, max(-Gap_i)).
        
        This requires calculating projections for all 12 axes for every pair.
        """
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        d_sq = dx*dx + dy*dy
        
        # Early exit for far apart
        if d_sq > 4.0:
            return 0.0 # Disjoint
        
        # Calculate projections
        # Axes for hex1 (rotated by a1) and hex2 (rotated by a2)
        # We need the 6 axes for each.
        # Actually, for convex polygons, checking the 6 axes of each is sufficient.
        
        gaps = []
        
        # Helper to get vertices quickly for projection
        # To save time in the inner loop, we might precompute cos/sin
        a1_rad = math.radians(a1)
        a2_rad = math.radians(a2)
        
        # Vertices of hex1 relative to c1
        v1 = []
        for i in range(6):
            theta = a1_rad + i * math.pi / 3
            v1.append((math.cos(theta), math.sin(theta)))
        # Vertices of hex2 relative to c2
        v2 = []
        for i in range(6):
            theta = a2_rad + i * math.pi / 3
            v2.append((math.cos(theta), math.sin(theta)))
        
        # Axes for hex1 (normals to edges)
        # Edge i is between v[i] and v[i+1]. Normal is perpendicular.
        # For a regular hexagon, normals are at angles a + 30 + i*60.
        for i in range(6):
            theta_n = a1_rad + math.pi/6 + i * math.pi / 3
            nx = math.cos(theta_n)
            ny = math.sin(theta_n)
            
            # Project v1 (centered at 0,0 effectively for relative calc? No, we need absolute projections)
            # Let's calculate absolute projections.
            # proj = dot(v, n)
            # For hex1 centered at c1: proj1 = dot(c1, n) + dot(v1_rel, n)
            # For hex2 centered at c2: proj2 = dot(c2, n) + dot(v2_rel, n)
            
            min1 = float('inf')
            max1 = float('-inf')
            for vx, vy in v1:
                p = (c1[0] + vx) * nx + (c1[1] + vy) * ny
                if p < min1: min1 = p
                if p > max1: max1 = p
                
            min2 = float('inf')
            max2 = float('-inf')
            for vx, vy in v2:
                p = (c2[0] + vx) * nx + (c2[1] + vy) * ny
                if p < min2: min2 = p
                if p > max2: max2 = p
                
            # Gap is the distance between the intervals
            # If max1 < min2, gap is min2 - max1
            # If max2 < min1, gap is min1 - max2
            # Otherwise, gap is negative (overlap)
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                # Overlap on this axis
                # We need the "penetration" depth.
                # If intervals overlap, the gap is negative.
                # The magnitude of overlap is min(max1, max2) - max(min1, min2)
                # But for SAT, if any axis has positive gap, they are disjoint.
                # If all axes have negative/zero gap, they are overlapping.
                # We want to maximize the minimum gap.
                # So we record the gap (which will be <= 0).
                gap = min(max1, max2) - max(min1, min2)
            
            gaps.append(gap)
            
        for i in range(6):
            theta_n = a2_rad + math.pi/6 + i * math.pi / 3
            nx = math.cos(theta_n)
            ny = math.sin(theta_n)
            
            min1 = float('inf')
            max1 = float('-inf')
            for vx, vy in v1:
                p = (c1[0] + vx) * nx + (c1[1] + vy) * ny
                if p < min1: min1 = p
                if p > max1: max1 = p
                
            min2 = float('inf')
            max2 = float('-inf')
            for vx, vy in v2:
                p = (c2[0] + vx) * nx + (c2[1] + vy) * ny
                if p < min2: min2 = p
                if p > max2: max2 = p
                
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = min(max1, max2) - max(min1, min2)
            
            gaps.append(gap)
            
        # The minimum gap is the "separation".
        # If min_gap > 0, disjoint.
        # We want to minimize the "violation" = max(0, -min_gap).
        # violation = smooth_max(0, -min_gap)
        # -min_gap = max(-gap)
        # So violation = smooth_max(0, max(-gap))
        
        neg_gaps = [-g for g in gaps]
        max_neg_gap = max(neg_gaps)
        
        # If max_neg_gap < 0, all gaps are positive -> disjoint -> violation 0
        if max_neg_gap < 0:
            return 0.0
            
        # Smooth max of 0 and max_neg_gap
        # We use a smooth max between 0 and the worst negative gap
        # Actually, smooth_max(0, max_neg_gap) with a reasonable beta
        beta = 20.0
        return smooth_max([0.0, max_neg_gap], beta)

    def evaluate_smooth(p, outer_angle_deg=0.0):
        """
        Objective function for SLSQP.
        p: [x1, y1, a1, ..., xN, yN, aN]
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        # Center the configuration to help the outer hex calculation
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        # 1. Overlap Penalty (Smooth)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                c1 = shifted_centers[i]
                c2 = shifted_centers[j]
                a1 = angles[i]
                a2 = angles[j]
                
                # Quick distance check
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
                d_sq = dx*dx + dy*dy
                if d_sq < 4.0: # Only check if close enough
                    pen = signed_distance_hexagons(c1, c2, a1, a2)
                    overlap_penalty += pen
        
        # Weight the overlap heavily to ensure feasibility
        # We want to minimize side length, but only if overlaps are ~0.
        # If overlaps are large, we want to push them apart.
        # Using a large weight ensures the optimizer prioritizes removing overlaps.
        overlap_cost = overlap_penalty * 100.0
        
        # 2. Outer Side Length
        # We need to calculate the side length of the outer hexagon containing all shifted hexagons.
        # This part is non-smooth due to the max operation, but SLSQP can handle it with finite differences
        # if we don't provide gradients. However, providing a smooth approximation helps.
        
        # For the outer hex, we can also use a smooth max of the projections.
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_projs = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            # Vertices of this hex
            a_rad = math.radians(a)
            for k in range(6):
                theta = a_rad + k * math.pi / 3
                vx = cx + math.cos(theta)
                vy = cy + math.sin(theta)
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    max_projs.append(proj)
        
        # Smooth max of all projections
        beta_outer = 10.0
        smooth_max_proj = smooth_max(max_projs, beta_outer)
        
        side = 2.0 * smooth_max_proj / SQRT3
        
        return overlap_cost + side

    # Initial configurations
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,0),(2,1)]
    coords5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)]
    
    configs = [coords1, coords2, coords3, coords4, coords5]
    
    best_params = None
    best_score = float('inf')
    
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Run SLSQP from multiple starting points
    for config in configs:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize parameters
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        
        # Center the initial config
        cx_avg = sum(params[i*3] for i in range(N)) / N
        cy_avg = sum(params[i*3+1] for i in range(N)) / N
        for i in range(N):
            params[i*3] -= cx_avg
            params[i*3+1] -= cy_avg
            
        try:
            # SLSQP is good for constrained problems, but we are using a penalty method.
            # We can also use COBYLA or Nelder-Mead if SLSQP struggles with the non-smoothness of the max.
            # Let's try SLSQP first as it uses gradient info (approximated).
            
            # To help SLSQP, we might want to bound the search space slightly or just let it run.
            # maxiter should be moderate to allow multiple restarts.
            
            res = minimize(evaluate_smooth, np.array(params), method='SLSQP', 
                           options={'maxiter': 100, 'ftol': 1e-5})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            # If SLSQP fails, try Nelder-Mead on the same start
            try:
                res = minimize(evaluate_smooth, np.array(params), method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # If no result, fallback to a simple grid or the first config
    if best_params is None:
        params = []
        for q, r in configs[0]:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        best_score = evaluate_smooth(best_params)

    # Final Validation and Outer Angle Optimization
    final_params = best_params.copy()
    
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])
        
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
    
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    valid = True
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
        # If the smooth optimization resulted in overlaps (which can happen if the penalty wasn't high enough
        # or the smooth approximation was too loose), we need to fix it.
        # We can run a quick local SA to resolve overlaps while trying to keep the side length low.
        # Or, we can just return the best valid config found during the process if we tracked it.
        # Since we didn't track intermediate valid states explicitly, let's do a quick repair.
        
        # Repair: Simple SA on the current params to remove overlaps
        current_params = final_params.copy()
        current_score = evaluate_smooth(current_params)
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        
        # Time limit for repair
        repair_deadline = time.time() + 20.0
        
        while T > T_min and time.time() < repair_deadline:
            new_params = current_params.copy()
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            da = random.gauss(0, 15.0)
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate_smooth(new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_params = new_params
                current_score = new_score
            
            T *= cooling_rate
        
        final_params = current_params
        # Re-center
        cx_avg = sum(final_params[i*3] for i in range(N)) / N
        cy_avg = sum(final_params[i*3+1] for i in range(N)) / N
        for i in range(N):
            final_params[i*3] -= cx_avg
            final_params[i*3+1] -= cy_avg
            
        # Re-validate
        centers = []
        angles = []
        for i in range(N):
            centers.append((final_params[i*3], final_params[i*3+1]))
            angles.append(final_params[i*3+2])
            
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
            
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            # Fallback to a known valid config (e.g. coords1)
            inner_list = []
            for q, r in coords1:
                x, y = axial_to_cartesian(q, r, spacing=1.0)
                inner_list.append([x, y, 0.0])
            cx_avg_fb = sum(p[0] for p in inner_list) / N
            cy_avg_fb = sum(p[1] for p in inner_list) / N
            inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
            hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
            best_side = calc_outer_side(hexagons_fb, 0.0)
            return (inner_list, (0.0, 0.0), best_side, 0.0)
            
        # Use the repaired hexagons
        shifted_centers = centers
    else:
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        # Ensure hexagons list corresponds to shifted centers
        hexagons = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)

    # Optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
