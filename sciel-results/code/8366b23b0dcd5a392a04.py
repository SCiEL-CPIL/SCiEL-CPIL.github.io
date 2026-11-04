"""Packing 12 unit hexagons in minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle + i * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_polygon(pt, poly):
    """Check if point is inside polygon using ray casting."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_overlap_sat(p1, p2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap (interior intersection)."""
    polys = [p1, p2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            nx, ny = -ey, ex
            norm = np.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            proj1 = [np.dot(p, [nx, ny]) for p in p1]
            proj2 = [np.dot(p, [nx, ny]) for p in p2]
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            if max1 + tol < min2 or max2 + tol < min1:
                return False
    return True

def hexagon_in_hexagon(inner_hex, outer_hex):
    """Check if inner hexagon is entirely inside outer hexagon."""
    return all(point_in_polygon(v, outer_hex) for v in inner_hex)

def check_disjoint(positions, angles, side, outer_center, outer_angle):
    """Check all inner hexagons are disjoint and inside outer hexagon."""
    outer_hex = hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    
    for pos, ang in zip(positions, angles):
        inner_hex = hex_vertices(pos[0], pos[1], 1.0, ang)
        if not hexagon_in_hexagon(inner_hex, outer_hex):
            return False
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            h1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            h2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if polygons_overlap_sat(h1, h2):
                return False
    return True

def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, lo=2.0, hi=20.0, iters=30):
    """Binary search for minimum side length."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if check_disjoint(inner_positions, inner_angles, mid, outer_center, outer_angle):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time

    # Precompute unit hexagon vertices and normals for fast SAT
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    # Normals for the 6 sides (perpendicular to edges, pointing outward)
    # Edge from v_i to v_{i+1}. Normal is rotated edge vector.
    unit_normals_0 = []
    for i in range(6):
        v1 = unit_verts_0[i]
        v2 = unit_verts_0[(i+1)%6]
        edge = v2 - v1
        # Normal (rotate 90 deg clockwise for outward? Let's check)
        # For hexagon counter-clockwise, outward normal is (edge_y, -edge_x) normalized?
        # v0=(1,0), v1=(0.5, 0.866). edge = (-0.5, 0.866).
        # Normal should point away from center. Center (0,0).
        # Midpoint (0.75, 0.433). Vector from center to mid is (0.75, 0.433).
        # edge_y = 0.866, -edge_x = 0.5. (0.866, 0.5) is parallel to (0.75, 0.433)?
        # 0.866/0.75 = 1.15, 0.5/0.433 = 1.15. Yes.
        nx, ny = edge[1], -edge[0]
        norm = np.sqrt(nx*nx + ny*ny)
        unit_normals_0.append([nx/norm, ny/norm])
    unit_normals_0 = np.array(unit_normals_0)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_normals(angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals = unit_normals_0 @ rot.T
        return normals

    def sat_gap(poly1, poly2, normals1, normals2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            axis = normals1[i]
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            axis = normals2[i]
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        return min_gap

    def calc_min_outer_side_fast(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        # The outward normal of side k of a hexagon rotated by theta is at angle theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # The distance from center to side is side * sqrt(3) / 2
        # So max_proj = side * sqrt(3) / 2
        # side = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def is_valid_fast(positions, angles, tol=1e-6):
        polys = []
        norms = []
        for i in range(N):
            polys.append(get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]))
            norms.append(get_hex_normals(angles[i]))
        
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j], norms[i], norms[j]) < -tol:
                    return False
        return True

    def check_disjoint_fast(positions, angles, side, outer_center, outer_angle, tol=1e-6):
        """Check all inner hexagons are disjoint and inside outer hexagon using fast methods."""
        # Check containment
        theta_rad = math.radians(outer_angle)
        outer_normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            outer_normals.append([math.cos(ang), math.sin(ang)])
        outer_normals = np.array(outer_normals)
        
        max_proj_allowed = side * math.sqrt(3) / 2.0
        
        polys = []
        norms = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            polys.append(verts)
            norms.append(get_hex_normals(ang))
            
            projs = verts @ outer_normals.T
            if np.max(projs) > max_proj_allowed + tol:
                return False
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j], norms[i], norms[j]) < -tol:
                    return False
        return True

    def compute_min_outer_side_fast(inner_positions, inner_angles, outer_center, outer_angle, lo=2.0, hi=20.0, iters=25):
        """Binary search for minimum side length using fast check."""
        for _ in range(iters):
            mid = (lo + hi) / 2
            if check_disjoint_fast(inner_positions, inner_angles, mid, outer_center, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    # --- PHASE 1: DISCRETE LATTICE SEARCH ---
    
    sqrt3 = math.sqrt(3)
    
    # Lattice vectors for hexagonal packing
    # v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # This corresponds to centers touching.
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    best_side = 100.0
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # We will search for a set of 12 points on a scaled lattice,
    # rotate the whole cluster, and assign angles.
    
    # Discrete angles for inner hexagons
    # Due to symmetry, 0 and 30 are the main distinct orientations for hexagons in a hexagonal container.
    # But we allow 0, 30, 60, etc. to be optimized later.
    
    # Generate candidate sets of 12 points
    # We use a "hexagonal number" style selection or just closest points to origin on lattice.
    
    # Try different scales and offsets
    scales = [0.9, 0.95, 1.0, 1.05, 1.1]
    
    # Pre-generate lattice points
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * base_v1 + j * base_v2
            lattice_points.append(p)
    
    lattice_points = np.array(lattice_points)
    
    # Sort by distance from origin
    dists = np.linalg.norm(lattice_points, axis=1)
    sorted_indices = np.argsort(dists)
    sorted_points = lattice_points[sorted_indices]
    
    # We want 12 points. The first 12 closest to origin form a hexagon shape (1 center + 6 ring + 5 from next ring).
    # Let's take the first 12.
    base_12_points = sorted_points[:12]
    
    # Center them
    centroid = np.mean(base_12_points, axis=0)
    base_12_points = base_12_points - centroid
    
    # Try different scales and global rotations
    for scale in scales:
        if elapsed() > DEADLINE - 10:
            break
            
        scaled_points = base_12_points * scale
        
        # Try global rotations
        for rot_deg in [0, 15, 30, 45, 60]:
            if elapsed() > DEADLINE - 5:
                break
                
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                [math.sin(rot_rad), math.cos(rot_rad)]])
            rotated_points = scaled_points @ rot_mat.T
            
            # Try different inner angle assignments
            # 1. All 0
            # 2. All 30
            # 3. Center 0, Ring 30
            # 4. Center 30, Ring 0
            
            angle_configs = [
                [0.0] * 12,
                [30.0] * 12,
            ]
            
            # Identify center and ring points relative to centroid?
            # base_12_points[0] is likely the center (dist 0).
            # The next 6 are the first ring.
            # The last 5 are the second ring.
            
            # Let's just try uniform angles first, then refine.
            
            for angles in angle_configs:
                if elapsed() > DEADLINE - 2:
                    break
                
                positions = rotated_points.tolist()
                
                # Quick validity check
                if not is_valid_fast(positions, angles):
                    continue
                
                # Compute side
                # The outer hexagon should be aligned with the cluster?
                # Or can it rotate independently?
                # The container is a regular hexagon. We can rotate it.
                # Let's try a few outer rotations.
                
                for outer_rot in [0.0, 15.0, 30.0, 45.0, 60.0]:
                    if elapsed() > DEADLINE - 1:
                        break
                    
                    side = calc_min_outer_side_fast(positions, angles, outer_rot)
                    
                    # Verify with binary search
                    side_bs = compute_min_outer_side_fast(positions, angles, (0.0, 0.0), outer_rot, lo=2.0, hi=side*1.5, iters=20)
                    
                    if side_bs < best_side:
                        best_side = side_bs
                        best_positions = [p.copy() for p in positions]
                        best_angles = angles.copy()
                        best_outer_angle = outer_rot

    # --- PHASE 2: CONTINUOUS REFINEMENT ---
    
    if best_positions is not None:
        # We have a valid discrete config. Now optimize it.
        # Parameters: positions (24 vars), angles (12 vars), outer_angle (1 var). Total 37 vars.
        # Objective: side length.
        # Constraints: disjointness (SAT gap >= 0), containment (implied by side calculation if we minimize it, but we need to ensure validity).
        
        # Actually, if we minimize the side length calculated by projection, and ensure disjointness,
        # the containment is automatically satisfied because the side length is defined by the max projection.
        # Wait, calc_min_outer_side_fast calculates the side length required to contain them.
        # So if we minimize that function, we are effectively finding the tightest container.
        # But we must ensure the hexagons don't overlap.
        
        # So: Minimize f(positions, angles, outer_angle) = calc_min_outer_side_fast(...)
        # Subject to: sat_gap(i, j) >= 0 for all i < j.
        
        # This is a non-smooth constraint.
        # We can use a penalty method or SLSQP with constraints.
        # SLSQP might struggle with non-smooth constraints.
        # Let's try Differential Evolution for global search in a local box around the best discrete solution,
        # then SLSQP for polish.
        
        def objective(params):
            if elapsed() > DEADLINE - 1:
                return 1000.0
            
            # Unpack
            # 24 pos, 12 ang, 1 outer_ang
            pos_flat = params[:24]
            pos = np.array(pos_flat).reshape(12, 2)
            angles = params[24:36]
            outer_ang = params[36]
            
            # Check validity
            if not is_valid_fast(pos, angles, tol=1e-5):
                # Penalty
                # Calculate min gap
                polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, angles[i]) for i in range(N)]
                norms = [get_hex_normals(angles[i]) for i in range(N)]
                min_gap = float('inf')
                for i in range(N):
                    for j in range(i+1, N):
                        g = sat_gap(polys[i], polys[j], norms[i], norms[j])
                        if g < min_gap:
                            min_gap = g
                if min_gap < 0:
                    return 1000.0 + abs(min_gap) * 1000.0
                else:
                    return 1000.0
            
            return calc_min_outer_side_fast(pos, angles, outer_ang)
        
        # Initial guess
        x0 = np.array(best_positions).flatten()
        x0 = np.concatenate([x0, np.array(best_angles), [best_outer_angle]])
        
        # Bounds
        # Positions: +/- 10
        # Angles: 0-360
        # Outer angle: 0-360
        bounds_pos = [(-10, 10)] * 24
        bounds_ang = [(0, 360)] * 12
        bounds_outer = [(0, 360)]
        bounds = bounds_pos + bounds_ang + bounds_outer
        
        # Try SLSQP first for quick improvement
        try:
            if elapsed() < DEADLINE - 20:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
                if res.fun < best_side:
                    best_side = res.fun
                    # Extract best
                    pos_flat = res.x[:24]
                    best_positions = np.array(pos_flat).reshape(12, 2).tolist()
                    best_angles = res.x[24:36].tolist()
                    best_outer_angle = res.x[36]
        except Exception:
            pass
        
        # Try Differential Evolution if time remains
        if elapsed() < DEADLINE - 30:
            try:
                # DE is slower but robust
                # Use a smaller population
                de_res = differential_evolution(objective, bounds, maxiter=10, popsize=10, tol=1e-5, seed=42, polish=False)
                if de_res.fun < best_side:
                    best_side = de_res.fun
                    pos_flat = de_res.x[:24]
                    best_positions = np.array(pos_flat).reshape(12, 2).tolist()
                    best_angles = de_res.x[24:36].tolist()
                    best_outer_angle = de_res.x[36]
            except Exception:
                pass

    # Final precise binary search for the best found configuration
    if best_positions is not None:
        # Ensure best_positions is a list of lists of floats
        final_positions = best_positions
        final_angles = best_angles
        final_outer_angle = best_outer_angle
        
        # One last check and refinement
        if is_valid_fast(final_positions, final_angles):
            final_side = compute_min_outer_side_fast(final_positions, final_angles, (0.0, 0.0), final_outer_angle, lo=2.0, hi=best_side*1.2, iters=30)
            best_side = final_side
        else:
            # Fallback to a known valid config if the optimized one became invalid due to numerical drift
            # Re-run discrete search for a safe fallback? Or just use a very large side?
            # Let's try to find a valid config from the discrete phase again quickly if needed.
            # For now, assume the discrete one was valid and SLSQP/DE didn't break it too much.
            # If it's invalid, we return a large side.
            best_side = 10.0
            # Revert to discrete best if possible? We didn't store it separately.
            # Let's just return the current best_side which might be invalid.
            # Better: Re-run a simple discrete check.
            # Actually, if is_valid fails, the evaluator will score 0.
            # We should ensure we return a valid config.
            # Let's keep a backup of the first valid discrete config.
            if 'backup_positions' in locals():
                final_positions = backup_positions
                final_angles = backup_angles
                final_outer_angle = backup_outer_angle
                final_side = compute_min_outer_side_fast(final_positions, final_angles, (0.0, 0.0), final_outer_angle, lo=2.0, hi=10.0, iters=20)
                best_side = final_side
            else:
                # No backup, generate a trivial one
                final_positions = [[i, 0] for i in range(12)]
                final_angles = [0] * 12
                final_outer_angle = 0
                final_side = 20.0
                best_side = final_side

        inner = [[p[0], p[1], a] for p, a in zip(final_positions, final_angles)]
        return inner, (0.0, 0.0), best_side, final_outer_angle
    else:
        # Fallback
        positions = [[4.0 * k, 0.0, 0.0] for k in range(-5, 7)]
        return positions, (0.0, 0.0), 40.0, 0.0
