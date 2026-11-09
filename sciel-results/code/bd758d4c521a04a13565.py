import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3.0)

    # --- Geometry Helpers ---

    def get_hex_vertices(cx, cy, angle_deg):
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return np.array([(cx + math.cos(a), cy + math.sin(a)) for a in angles])

    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def dist_point_to_segment(p, a, b):
        # p, a, b are (x, y) tuples or arrays
        px, py = p
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def min_dist_polygons(v1, v2):
        """
        Compute minimum distance between two convex polygons.
        Returns 0 if they overlap or touch.
        """
        # Check containment
        for v in v1:
            if point_in_polygon(v, v2):
                return 0.0
        for v in v2:
            if point_in_polygon(v, v1):
                return 0.0
        
        # Check edge-edge distances
        min_d = float('inf')
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                
                # Distance from p1 to segment p3-p4
                d1 = dist_point_to_segment(p1, p3, p4)
                if d1 < min_d: min_d = d1
                
                # Distance from p3 to segment p1-p2
                d2 = dist_point_to_segment(p3, p1, p2)
                if d2 < min_d: min_d = d2
                
                # Check for segment intersection (optional if dist is 0)
                # If segments intersect, dist is 0
                # Simple check: if d1 == 0 or d2 == 0, they touch/overlap
                if min_d == 0.0:
                    return 0.0
        return min_d

    def compute_L(params):
        """
        Compute the required outer side length L.
        params: array of shape (N, 3) -> [x, y, angle_deg]
        """
        max_proj = 0.0
        # Normals for flat-top outer hexagon
        normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
        normal_cos = np.array([math.cos(a) for a in normal_angles])
        normal_sin = np.array([math.sin(a) for a in normal_angles])
        
        for i in range(N):
            x, y, ang = params[i]
            verts = get_hex_vertices(x, y, ang)
            # Project all vertices onto normals
            # verts: (6, 2)
            # projs: (6, 6)
            projs = verts[:, 0:1] * normal_cos[None, :] + verts[:, 1:2] * normal_sin[None, :]
            curr_max = np.max(projs)
            if curr_max > max_proj:
                max_proj = curr_max
        return max_proj / (sqrt3 / 2.0)

    def compute_penalty(params, delta=1e-5):
        """
        Compute smooth penalty for overlaps.
        """
        penalty = 0.0
        verts_list = []
        for i in range(N):
            x, y, ang = params[i]
            verts = get_hex_vertices(x, y, ang)
            verts_list.append(verts)
        
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_polygons(verts_list[i], verts_list[j])
                if d < delta:
                    # Smooth penalty: (delta - d)^2
                    penalty += (delta - d) ** 2
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        L = compute_L(params)
        pen = compute_penalty(params)
        # Weight for penalty. We want L to dominate, but prevent overlap.
        # If pen is 0, we minimize L.
        # If pen > 0, we add a cost.
        return L + 100.0 * pen

    # --- Phase 1: Combinatorial Grid Search (from seed) ---
    
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))

    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta):
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        best_L = float('inf')
        best_theta = 0.0
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine
        for _ in range(2):
            step = 0.5 if _ == 0 else 0.1
            t_start = best_theta - step * 6
            t_end = best_theta + step * 6
            for t_deg in np.arange(t_start, t_end + step * 0.5, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    best_L_comb = float('inf')
    best_params_comb = None

    cluster_angles_coarse = [math.radians(a) for a in range(0, 61, 5)]
    if math.radians(30) not in cluster_angles_coarse:
        cluster_angles_coarse.append(math.radians(30))

    for cluster_angle in cluster_angles_coarse:
        if time.time() > deadline - 60.0: # Reserve time for optimization
            break

        cos_ca = math.cos(cluster_angle)
        sin_ca = math.sin(cluster_angle)

        rotated_centers = []
        for x, y in all_centers:
            rx = x * cos_ca - y * sin_ca
            ry = x * sin_ca + y * cos_ca
            rotated_centers.append((rx, ry))

        center_optimal = []
        for i in range(19):
            x, y = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x, y)
            center_optimal.append((L_h, theta_h))

        for subset in combinations(range(19), 12):
            if time.time() > deadline - 60.0:
                break
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L_comb:
                best_L_comb = L
                # Store params for this subset
                params = np.zeros((N, 3))
                for idx, i in enumerate(subset):
                    x, y = rotated_centers[i]
                    params[idx, 0] = x
                    params[idx, 1] = y
                    params[idx, 2] = math.degrees(center_optimal[i][1])
                best_params_comb = params

    if best_params_comb is None:
        # Fallback
        best_params_comb = np.zeros((N, 3))
        for i in range(N):
            x, y = all_centers[i]
            best_params_comb[i, 0] = x
            best_params_comb[i, 1] = y
            best_params_comb[i, 2] = 0.0
        best_L_comb = compute_L(best_params_comb)

    # --- Phase 2: Continuous Optimization ---
    
    if time.time() < deadline - 10.0:
        try:
            # Use Nelder-Mead for robustness
            res = minimize(
                objective, 
                best_params_comb, 
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500, 'maxfev': 1000}
            )
            
            if res.success or res.fun < best_L_comb:
                best_params_opt = res.x
                best_L_opt = res.fun
            else:
                best_params_opt = best_params_comb
                best_L_opt = best_L_comb
        except Exception as e:
            # Fallback to combinatorial result if optimization fails
            best_params_opt = best_params_comb
            best_L_opt = best_L_comb

        # Post-processing: Ensure strict disjointness
        # If the optimizer found a solution with overlaps (penalty > 0), 
        # we might want to scale out slightly or just accept if penalty is small.
        # The penalty term 100*pen ensures that if there is overlap, the objective is high.
        # So best_L_opt should correspond to a non-overlapping config.
        
        # However, due to numerical precision, there might be tiny overlaps.
        # We can check the final min distance.
        final_pen = compute_penalty(best_params_opt, delta=0.0)
        if final_pen > 0:
            # There is overlap. Scale centers out from origin to fix.
            # This increases L.
            # Simple fix: multiply coordinates by 1.001
            best_params_opt[:, 0:2] *= 1.001
            best_L_opt = compute_L(best_params_opt)
            
    else:
        best_params_opt = best_params_comb
        best_L_opt = best_L_comb

    # Construct return values
    inner = []
    for i in range(N):
        x, y, ang = best_params_opt[i]
        inner.append([float(x), float(y), float(ang)])
    
    # The outer hexagon is centered at (0,0) with side length best_L_opt and rotation 0
    # Wait, the problem asks for outer center, side, angle.
    # The computation assumed outer center (0,0) and flat-top (angle 0).
    # We can return (0,0), best_L_opt, 0.0.
    
    return inner, (0.0, 0.0), float(best_L_opt), 0.0

# EVOLVE-BLOCK-END
