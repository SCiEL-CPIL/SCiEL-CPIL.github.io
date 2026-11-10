import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

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

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

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
        
        # Check edge-edge intersection
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return 0.0
        
        # Check edge-edge distances
        min_d = float('inf')
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
                
                if min_d == 0.0:
                    return 0.0
        return min_d

    def compute_L(params, center=(0.0, 0.0)):
        """
        Compute the required outer side length L for a flat-top hexagon centered at (0,0).
        params: array of shape (N, 3) -> [x, y, angle_deg]
        """
        max_proj = 0.0
        # Normals for flat-top outer hexagon
        normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
        normal_cos = np.array([math.cos(a) for a in normal_angles])
        normal_sin = np.array([math.sin(a) for a in normal_angles])
        
        for i in range(N):
            x, y, ang = params[i]
            # Translate to center relative to outer center
            x_rel = x - center[0]
            y_rel = y - center[1]
            verts = get_hex_vertices(x_rel, y_rel, ang)
            # Project all vertices onto normals
            projs = verts[:, 0:1] * normal_cos[None, :] + verts[:, 1:2] * normal_sin[None, :]
            curr_max = np.max(projs)
            if curr_max > max_proj:
                max_proj = curr_max
        return max_proj / (sqrt3 / 2.0)

    def compute_penalty(params, delta=0.0):
        """
        Compute penalty for overlaps.
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
                    # Strong penalty for overlap
                    penalty += (delta - d) ** 2
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        L = compute_L(params)
        # Strong penalty weight
        pen = compute_penalty(params, delta=0.0)
        return L + 1000.0 * pen

    # --- Initial Configuration: Valid Flower Packing ---
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance 2*apothem = sqrt(3) from center?
    # Distance between centers of touching hexagons is 2 * inradius = sqrt(3).
    # Inradius of unit hexagon = sqrt(3)/2.
    # So neighbors are at distance sqrt(3).
    
    init_params = np.zeros((N, 3))
    
    # Center
    init_params[0] = [0.0, 0.0, 0.0]
    
    # 6 Neighbors
    for k in range(6):
        angle = math.radians(60 * k)
        r = sqrt3
        init_params[k+1, 0] = r * math.cos(angle)
        init_params[k+1, 1] = r * math.sin(angle)
        init_params[k+1, 2] = 0.0
        
    # 5 more from Ring 2 (to make 12 total)
    # Ring 2 centers are at distance 2*sqrt(3) from origin?
    # Actually, let's just place them at valid non-overlapping spots.
    # The "flower" of 7 is tight. We need 5 more.
    # Let's place them at angles 30, 90, 150, 210, 270 degrees at radius 2*sqrt(3) * something?
    # Simpler: Just use a known valid grid.
    # Let's try placing them in a larger ring.
    # Distance from center to ring 2 vertices in a hex grid is 2*sqrt(3).
    # Let's place 5 hexagons at radius 2*sqrt(3) at angles 30, 90, 150, 210, 270.
    for k in range(5):
        angle = math.radians(30 + 60 * k)
        r = 2 * sqrt3
        init_params[k+7, 0] = r * math.cos(angle)
        init_params[k+7, 1] = r * math.sin(angle)
        init_params[k+7, 2] = 0.0

    # Verify initial validity
    init_pen = compute_penalty(init_params, delta=0.0)
    if init_pen > 0:
        # Fallback: spread them out more to ensure no overlap, then optimize
        for i in range(N):
            init_params[i, 0] *= 1.2
            init_params[i, 1] *= 1.2

    best_params = init_params.copy()
    best_L = compute_L(best_params)

    # --- Optimization ---
    if time.time() < deadline - 20.0:
        try:
            # Use Nelder-Mead for local refinement
            # We need to ensure the optimizer doesn't increase the penalty.
            # We'll run multiple restarts if time permits, but for now, one good run.
            
            # To improve robustness, we can fix the center of mass to (0,0) to reduce degrees of freedom,
            # but Nelder-Mead on 36 dims is slow. 
            # Let's try optimizing positions only, keeping angles at 0 initially, then optimize angles.
            
            # Phase 2a: Optimize positions (angles fixed to 0)
            def objective_pos(params_2d):
                if time.time() > deadline:
                    return 1e10
                params = np.zeros((N, 3))
                params[:, 0:2] = params_2d
                params[:, 2] = 0.0
                return objective(params)
            
            x0_pos = best_params[:, 0:2].flatten()
            res_pos = minimize(
                objective_pos, 
                x0_pos, 
                method='Nelder-Mead',
                options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500, 'maxfev': 1000}
            )
            
            if res_pos.fun < best_L:
                best_params[:, 0:2] = res_pos.x.reshape(N, 2)
                best_L = compute_L(best_params)
                # Recalculate penalty to ensure validity
                if compute_penalty(best_params, delta=0.0) > 0:
                    # If invalid, revert to previous valid state
                    best_params[:, 0:2] = x0_pos.reshape(N, 2)
                    best_L = compute_L(best_params)

            # Phase 2b: Optimize angles (positions fixed)
            def objective_ang(params_1d):
                if time.time() > deadline:
                    return 1e10
                params = best_params.copy()
                params[:, 2] = params_1d
                return objective(params)
            
            x0_ang = best_params[:, 2].copy()
            res_ang = minimize(
                objective_ang, 
                x0_ang, 
                method='Nelder-Mead',
                options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500, 'maxfev': 1000}
            )
            
            if res_ang.fun < best_L:
                best_params[:, 2] = res_ang.x
                # Check validity
                if compute_penalty(best_params, delta=0.0) <= 1e-6:
                    best_L = compute_L(best_params)
                else:
                    # Revert angles
                    best_params[:, 2] = x0_ang

            # Phase 2c: Joint optimization (if time)
            if time.time() < deadline - 10.0:
                res_joint = minimize(
                    objective, 
                    best_params.flatten(), 
                    method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 300, 'maxfev': 500}
                )
                
                if res_joint.fun < best_L:
                    candidate = res_joint.x.reshape(N, 3)
                    if compute_penalty(candidate, delta=0.0) <= 1e-6:
                        best_params = candidate
                        best_L = compute_L(best_params)

        except Exception as e:
            pass # Keep current best

    # Final Check and Correction
    final_pen = compute_penalty(best_params, delta=0.0)
    if final_pen > 1e-6:
        # If still overlapping, scale out from centroid
        centroid = np.mean(best_params[:, 0:2], axis=0)
        for i in range(N):
            dx = best_params[i, 0] - centroid[0]
            dy = best_params[i, 1] - centroid[1]
            # Scale factor to ensure distance > min required
            # Simple heuristic: scale by 1.01
            best_params[i, 0] = centroid[0] + dx * 1.01
            best_params[i, 1] = centroid[1] + dy * 1.01
        best_L = compute_L(best_params)

    # Construct return values
    inner = []
    for i in range(N):
        x, y, ang = best_params[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_L), 0.0

# EVOLVE-BLOCK-END
