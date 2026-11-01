import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    # Existing candidates from current program
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # New Compact Candidates
    # BFS Ball: Collect all axial coords, sort by distance, take first 11
    def generate_bfs_ball(n_hexes):
        visited = set()
        queue = [(0, 0)]
        visited.add((0, 0))
        coords = []
        
        # Hex directions in axial
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        while len(coords) < n_hexes and queue:
            q, r = queue.pop(0)
            coords.append((q, r))
            for dq, dr in dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    queue.append((nq, nr))
        return coords

    # The BFS above is simple BFS, which creates a "diamond" shape in axial coords.
    # We want a more circular "ball".
    # Let's generate all coords within radius 2 and pick the 11 with smallest distance.
    def generate_compact_ball(n_hexes):
        coords = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                # Axial distance: (|q| + |r| + |q+r|) / 2
                dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
                if dist <= 2.0:
                    coords.append((q, r, dist))
        
        # Sort by distance, then by angle for symmetry
        coords.sort(key=lambda x: (x[2], x[0], x[1]))
        return [ (q, r) for q, r, d in coords[:n_hexes] ]

    candidates.append(generate_compact_ball(11))
    
    # Another variant: 3x3 grid in axial
    axial_13 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    # This is same as axial_3. Let's try a different compact one.
    # Center + 6 neighbors + 4 more from the next ring
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    # This is axial_2.
    
    # Let's add a "staggered" row candidate
    axial_15 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2)
    ]
    candidates.append(axial_15)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    def optimize_positions_and_container(pts_init, beta_init):
        """
        Optimize the positions of the hexagons and the container simultaneously.
        This allows the 'grid' to distort to reduce the bounding box.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None
        
        n_pts = len(pts_init)
        # Variables: x1, y1, ..., xn, yn, cx, cy, beta
        # We fix beta to beta_init initially to simplify, or optimize it too?
        # Let's optimize beta too.
        # Total vars: 2*n_pts + 3
        
        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
                
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            
            # Normalize beta
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            # 1. Calculate Container S
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            max_proj = 0.0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
            S = (max_proj + APOTHEM) / APOTHEM
            
            # 2. Calculate Overlap Penalty
            # Distance between centers must be >= 2 * APOTHEM (for non-rotated hexes)
            # For rotated hexes, the minimum distance is more complex.
            # Approximation: The "radius" of a hexagon is APOTHEM (distance to side)
            # and the "vertex radius" is 1.0.
            # The minimum distance between centers of two disjoint hexagons is at least 2*APOTHEM.
            # If they rotate, they might touch at vertices, requiring distance 2.0?
            # Actually, the support function approach is better, but for a quick penalty:
            # If dist < 2*APOTHEM, they definitely overlap.
            # If dist < 2.0, they might overlap depending on angle.
            # Let's use a soft penalty for dist < 2*APOTHEM.
            
            penalty = 0.0
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = 2 * APOTHEM
                    if dist < min_dist:
                        # Quadratic penalty
                        penalty += (min_dist - dist) ** 2
            
            # Weight the penalty heavily
            return S + 100.0 * penalty

        # Initial guess
        init_params = np.concatenate([pts_init.flatten(), [0.0, 0.0, beta_init]])
        
        try:
            # Use Nelder-Mead as it handles non-smooth penalties well
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < 1e9:
                best_params = res.x
                pts_opt = best_params[:2*n_pts].reshape(n_pts, 2)
                cx = best_params[2*n_pts]
                cy = best_params[2*n_pts+1]
                beta = best_params[2*n_pts+2]
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                # Verify with exact calc
                S_exact, c_exact, D_exact = calc_S_for_beta(pts_opt, beta)
                # Check for overlaps in final solution
                valid = True
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        dx = pts_opt[i, 0] - pts_opt[j, 0]
                        dy = pts_opt[i, 1] - pts_opt[j, 1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 2 * APOTHEM - 1e-5:
                            valid = False
                
                if valid:
                    return S_exact, pts_opt, (cx, cy), beta
                else:
                    # If invalid, return the original
                    return float('inf'), None, None, None
            else:
                return float('inf'), None, None, None
        except Exception:
            return float('inf'), None, None, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # First pass: Evaluate fixed grid candidates
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Second pass: Try to polish the top candidates by moving positions
    # We will try the top 3 candidates from the first pass
    # To avoid re-running all, we just take the best one found so far and try to polish it.
    # Also, we should try polishing the "Compact Ball" specifically.
    
    top_candidates_to_polish = []
    # Add the global best
    if best_global_pts is not None:
        top_candidates_to_polish.append(best_global_pts.copy())
    
    # Add the BFS ball specifically if it wasn't the best
    bfs_ball = generate_compact_ball(11)
    bfs_pts = axial_to_cartesian(bfs_ball)
    S_bfs, beta_bfs, center_bfs = optimize_shape(bfs_pts)
    if S_bfs < best_global_S:
        best_global_S = S_bfs
        best_global_pts = bfs_pts
        best_global_beta = beta_bfs
        best_global_center = center_bfs
        top_candidates_to_polish = [bfs_pts.copy()]
    else:
        # If bfs was not best, add it to polish list if time permits
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            top_candidates_to_polish.append(bfs_pts.copy())
            
    # Polish the best candidate
    if top_candidates_to_polish:
        for pts_init in top_candidates_to_polish:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            S_polished, pts_polished, center_polished, beta_polished = optimize_positions_and_container(pts_init, best_global_beta)
            
            if S_polished < best_global_S:
                best_global_S = S_polished
                best_global_pts = pts_polished
                best_global_beta = beta_polished
                best_global_center = center_polished

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale to ensure disjointness (safety)
    # The optimization used a penalty, so it might be slightly overlapping.
    # We scale the positions relative to the center to ensure minimum distance.
    # Actually, if we scale, we change the shape.
    # Better: check min distance. If < 2*APOTHEM, scale up.
    min_dist_found = float('inf')
    for i in range(len(best_global_pts)):
        for j in range(i+1, len(best_global_pts)):
            dx = best_global_pts[i, 0] - best_global_pts[j, 0]
            dy = best_global_pts[i, 1] - best_global_pts[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist_found = min(min_dist_found, dist)
            
    required_dist = 2 * APOTHEM
    if min_dist_found < required_dist:
        scale_factor = required_dist / min_dist_found
        # Scale positions relative to center
        for i in range(len(best_global_pts)):
            best_global_pts[i, 0] = cx + (best_global_pts[i, 0] - cx) * scale_factor
            best_global_pts[i, 1] = cy + (best_global_pts[i, 1] - cy) * scale_factor
        
        # Recalculate S
        max_proj = 0
        for k in range(6):
            ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
