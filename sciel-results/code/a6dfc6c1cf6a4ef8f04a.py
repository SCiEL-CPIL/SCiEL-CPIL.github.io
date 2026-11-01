"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def convex_hull(points):
        """Compute convex hull of a set of 2D points."""
        points = np.array(points)
        n = len(points)
        if n <= 1:
            return points
        
        # Sort by x, then y
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
                upper.pop()
            upper.append(p)
        
        # Concatenate, removing duplicate endpoints
        return np.vstack([lower[:-1], upper[:-1]])

    def minkowski_sum_hex(v1, v2):
        """
        Compute Minkowski sum of two convex polygons.
        v1, v2: Nx2 arrays.
        Returns vertices of the Minkowski sum (convex hull of v1_i + v2_j).
        """
        sums = []
        for p1 in v1:
            for p2 in v2:
                sums.append(p1 + p2)
        return convex_hull(sums)

    def point_in_convex_poly_dist(px, py, poly):
        """
        Returns 0 if point is outside poly.
        Returns distance to boundary if point is inside poly.
        """
        # Check if inside using cross products
        inside = True
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            # Normal pointing outward (assuming CCW order)
            # For CCW, cross(edge, point - p1) should be > 0 for inside
            cross = edge[0] * (py - p1[1]) - edge[1] * (px - p1[0])
            if cross < 0:
                inside = False
                break
        
        if not inside:
            return 0.0
        
        # Calculate min distance to edges
        min_dist = float('inf')
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Distance from point to segment p1-p2
            v = p2 - p1
            w = np.array([px, py]) - p1
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(w)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(w - v)
                else:
                    t = c1 / c2
                    d = np.linalg.norm(w - t * v)
            if d < min_dist:
                min_dist = d
        
        return min_dist

    def objective(params):
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = params[3*n_inner+2]
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = max(os, 0.1)
        outer_angle = oa
        
        # Precompute outer vertices
        outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
        
        penalty = 0.0
        
        # 1. Inside penalty: Inner vertices must be inside outer hexagon
        # We use the distance from vertex to outer boundary if outside.
        # Note: point_in_convex_poly_dist returns 0 if outside, dist if inside.
        # We want the OPPOSITE: 0 if inside, dist if outside.
        # So we need a helper or modify the logic.
        
        def point_to_convex_poly_dist_outside(px, py, poly):
            # Returns 0 if inside, distance to boundary if outside.
            # First check if inside
            inside = True
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                cross = edge[0] * (py - p1[1]) - edge[1] * (px - p1[0])
                if cross < 0:
                    inside = False
                    break
            
            if inside:
                return 0.0
            
            # Calculate min distance to edges
            min_dist = float('inf')
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                v = p2 - p1
                w = np.array([px, py]) - p1
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(w)
                else:
                    c2 = np.dot(v, v)
                    if c2 <= c1:
                        d = np.linalg.norm(w - v)
                    else:
                        t = c1 / c2
                        d = np.linalg.norm(w - t * v)
                if d < min_dist:
                    min_dist = d
            return min_dist

        for i in range(n_inner):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                d = point_to_convex_poly_dist_outside(vx, vy, outer_verts)
                penalty += d * d

        # 2. Disjoint penalty: Inner hexagons must be disjoint.
        # Use Minkowski sum approach.
        # For pair (i, j), compute M = H_i + (-H_j).
        # Check if c_j - c_i is inside M.
        # If inside, penalty += dist(c_j - c_i, boundary(M))^2.
        
        # Precompute inner vertices and negative vertices
        inner_verts_list = []
        neg_inner_verts_list = []
        for i in range(n_inner):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            inner_verts_list.append(verts)
            # -H_j is H_j rotated 180 degrees, centered at origin?
            # Minkowski sum of H_i and -H_j.
            # -H_j vertices are - (H_j vertices - c_j) + c_j? No.
            # We want the shape -H_j.
            # If H_j is centered at c_j, the shape is centered at c_j.
            # The shape -H_j is centered at -c_j? No, the shape is just inverted.
            # Let's define H_j_local = H_j - c_j.
            # -H_j_local = - (H_j - c_j).
            # M = H_i_local + (-H_j_local).
            # Condition: c_j - c_i is NOT in M + c_i?
            # Actually, disjointness of H_i (centered c_i) and H_j (centered c_j)
            # is equivalent to c_j - c_i not being in H_i + (-H_j) (where H_i, H_j are centered at 0).
            
            # Let's compute H_i_centered and H_j_centered.
            # H_i_c = verts - c_i
            # H_j_c = verts_j - c_j
            # M = H_i_c + (-H_j_c)
            # Check if (c_j - c_i) is in M.
            
            # We will compute this in the loop to save memory if needed, but N=12 is small.
            pass

        # Recompute centered vertices for Minkowski
        centered_verts = []
        for i in range(n_inner):
            cx, cy, angle = inner_positions[i]
            verts = inner_verts_list[i]
            centered_verts.append(verts - np.array([cx, cy]))

        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                # M = H_i_centered + (-H_j_centered)
                # -H_j_centered is just the reflection of H_j_centered through origin.
                neg_j_verts = -centered_verts[j]
                
                # Minkowski sum
                M_verts = minkowski_sum_hex(centered_verts[i], neg_j_verts)
                
                # Vector from c_i to c_j
                ci = np.array([inner_positions[i][0], inner_positions[i][1]])
                cj = np.array([inner_positions[j][0], inner_positions[j][1]])
                rel_vec = cj - ci
                
                # Check if rel_vec is inside M_verts
                d = point_in_convex_poly_dist(rel_vec[0], rel_vec[1], M_verts)
                penalty += d * d

        # Total objective: Outer side + Penalty
        # We want to minimize outer_side.
        # If penalty is 0, return outer_side.
        # If penalty > 0, return outer_side + penalty * weight.
        # To encourage valid solutions, we can use a large weight.
        # However, L-BFGS-B minimizes the sum.
        # If we just return outer_side + penalty, the optimizer might try to reduce penalty by shrinking outer_side, which is bad.
        # We want to find a valid configuration for the smallest possible outer_side.
        
        # Strategy:
        # If valid (penalty < eps), return outer_side.
        # If invalid, return outer_side + penalty * 1000.
        
        if penalty < 1e-6:
            return outer_side
        else:
            return outer_side + penalty * 100.0

    # Initial valid configuration: Hexagonal lattice
    # This is a good starting point for hexagons.
    # Spacing in x: 2.0 (side * 2)
    # Spacing in y: sqrt(3) (side * sqrt(3))
    # Row offset: 1.0
    
    initial_positions = []
    # 3 rows of 4
    for i in range(3):
        y = i * np.sqrt(3)
        for j in range(4):
            x = (j - 1.5) * 2.0 + (i % 2) * 1.0
            initial_positions.append([x, y, 0.0])
    
    # Outer hexagon
    # Bounding box of the lattice
    # x: -3.5 to 3.5 (approx)
    # y: 0 to 2*sqrt(3) ~ 3.46
    # Center approx (0, 1.73)
    # Side length approx 4.0
    
    outer_side_init = 4.5
    outer_center_init = [0.0, np.sqrt(3)]
    outer_angle_init = 0.0
    
    params = []
    for pos in initial_positions:
        params.extend([pos[0], pos[1], pos[2]])
    params.extend([outer_center_init[0], outer_center_init[1], outer_side_init, outer_angle_init])
    
    best_params = params.copy()
    best_score = objective(params)
    
    # Optimize with L-BFGS-B
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Second pass: finer
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    final_params = best_params
    
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append([x, y, a])
    
    ox = final_params[3*N]
    oy = final_params[3*N+1]
    os = max(final_params[3*N+2], 0.1)
    oa = final_params[3*N+3]
    
    # Final validation check
    # Re-implement simple check for safety
    def simple_check(inner, outer_center, outer_side, outer_angle):
        # Inside
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        for i in range(len(inner)):
            cx, cy, angle = inner[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                # Check if inside outer
                # Using point_in_convex_poly_dist_outside
                d = point_to_convex_poly_dist_outside(vx, vy, outer_verts)
                if d > 1e-6:
                    return False
        
        # Disjoint
        for i in range(len(inner)):
            for j in range(i+1, len(inner)):
                ci = np.array([inner[i][0], inner[i][1]])
                cj = np.array([inner[j][0], inner[j][1]])
                ai = inner[i][2]
                aj = inner[j][2]
                
                vi = hexagon_vertices(ci[0], ci[1], 1.0, ai) - ci
                vj = hexagon_vertices(cj[0], cj[1], 1.0, aj) - cj
                
                M_verts = minkowski_sum_hex(vi, -vj)
                rel_vec = cj - ci
                d = point_in_convex_poly_dist(rel_vec[0], rel_vec[1], M_verts)
                if d > 1e-6:
                    return False
        return True

    if not simple_check(inner, (ox, oy), os, oa):
        # Fallback to initial valid guess
        inner = [[p[0], p[1], p[2]] for p in initial_positions]
        ox, oy = outer_center_init
        os = outer_side_init
        oa = outer_angle_init
        for _ in range(50):
            if simple_check(inner, (ox, oy), os, oa):
                break
            os += 0.1
            if os > 12.0:
                break

    return inner, (ox, oy), os, oa
# EVOLVE-BLOCK-END
