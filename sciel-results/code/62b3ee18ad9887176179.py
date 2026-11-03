import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Pre-compute hexagon geometry
    # Vertices of a unit regular hexagon centered at origin, angle 0
    # Angle 0 means one vertex is at (1, 0)? Or edge is horizontal?
    # Standard: vertices at angles 0, 60, 120, 180, 240, 300.
    # Let's use angles 0, 60... for vertices.
    # Then normals (axes for SAT) are at 30, 90, 150, 210, 270, 330.
    
    hex_verts = np.array([
        [math.cos(math.pi * i / 3.0), math.sin(math.pi * i / 3.0)] for i in range(6)
    ])
    
    # Normals for SAT
    sat_axes = np.array([
        [math.cos(math.pi / 6.0 + math.pi * i / 3.0), math.sin(math.pi / 6.0 + math.pi * i / 3.0)] for i in range(6)
    ])
    
    # Pre-compute projections of hex_verts onto sat_axes
    # proj[i, k] is the projection of vertex i onto axis k
    # Since hex_verts is fixed, this is constant.
    hex_projs = hex_verts @ sat_axes.T  # (6, 6)
    hex_min_projs = np.min(hex_projs, axis=0)  # (6,)
    hex_max_projs = np.max(hex_projs, axis=0)  # (6,)
    
    def check_collision(c1, c2):
        """
        Check if two unit hexagons at c1 and c2 (same orientation) collide.
        Returns True if they overlap.
        """
        diff = c2 - c1
        # Project centers onto axes
        center_projs = diff @ sat_axes.T  # (6,)
        
        # For each axis k:
        # Interval 1: [c1_k + min_k, c1_k + max_k]
        # Interval 2: [c2_k + min_k, c2_k + max_k]
        # Relative interval: [center_proj_k + min_k, center_proj_k + max_k]
        # Wait, it's easier to check if the distance between centers along axis k
        # is greater than the sum of the half-widths?
        # No, because the hexagon is not a circle.
        
        # Correct SAT check:
        # For axis k, let p1 = c1 @ axis_k, p2 = c2 @ axis_k.
        # The hexagon 1 occupies [p1 + min_k, p1 + max_k].
        # The hexagon 2 occupies [p2 + min_k, p2 + max_k].
        # They overlap on this axis if:
        # p1 + min_k < p2 + max_k  AND  p2 + min_k < p1 + max_k
        
        # Let d = p2 - p1 = center_projs[k]
        # Condition for NO overlap on axis k:
        # p1 + max_k <= p2 + min_k  OR  p2 + max_k <= p1 + min_k
        # d >= max_k - min_k  OR  d <= min_k - max_k
        # Note: max_k - min_k is the width of the hexagon along axis k.
        
        widths = hex_max_projs - hex_min_projs
        # d is center_projs
        # No overlap if any axis has |d| >= width
        # Wait, the condition is:
        # p2 + min_k >= p1 + max_k  => d >= max_k - min_k
        # p1 + min_k >= p2 + max_k  => d <= min_k - max_k = -width
        
        # So if for ANY k, |center_projs[k]| >= widths[k], they are disjoint.
        # If for ALL k, |center_projs[k]| < widths[k], they might overlap (need to check all).
        # Actually, if they are disjoint, there must be at least one separating axis.
        # So if we find one k where |d_k| >= width_k, they are disjoint.
        
        for k in range(6):
            if abs(center_projs[k]) >= widths[k]:
                return False # Disjoint
        
        return True # Overlapping

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
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

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def optimize_outer(pts):
        """Find best beta and center for fixed pts."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Grid search
        n_grid = 50
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        if best_center is None:
            return float('inf'), 0.0, None

        # Refine
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1)
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded', options={'maxiter': 100})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Generate Candidates
    candidates = []
    # Add original 30
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    
    # Add a few more diverse ones
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(2,2),(3,2),(4,2)]) # Long strip
    candidates.append([(0,0),(0,1),(0,2),(0,3),(0,4),(1,1),(1,2),(1,3),(2,2),(2,3),(3,2)]) # Blocky
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Local Relaxation
        # 1. Optimize outer
        S, beta, center = optimize_outer(pts)
        
        if center is None:
            continue
            
        # 2. Relax inner positions
        # We will try to move each point slightly to reduce S
        # We keep the best S and corresponding pts
        current_S = S
        current_beta = beta
        current_center = center
        current_pts = pts.copy()
        
        # Relaxation steps
        # Use a small step size
        step_size = 0.1
        max_relax_iters = 20
        
        for r_iter in range(max_relax_iters):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            improved = False
            for i in range(N):
                # Try moving point i
                original_pos = current_pts[i].copy()
                # Try 8 directions
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = original_pos + np.array([dx, dy])
                        current_pts[i] = new_pos
                        
                        # Check collision
                        collides = False
                        for j in range(N):
                            if i == j:
                                continue
                            if check_collision(current_pts[i], current_pts[j]):
                                collides = True
                                break
                        
                        if collides:
                            current_pts[i] = original_pos
                            continue
                        
                        # Optimize outer for new pts
                        new_S, new_beta, new_center = optimize_outer(current_pts)
                        
                        if new_center is not None and new_S < current_S:
                            current_S = new_S
                            current_beta = new_beta
                            current_center = new_center
                            improved = True
                        else:
                            current_pts[i] = original_pos
                            
            if not improved:
                break

        if current_S < best_global_S:
            best_global_S = current_S
            best_global_pts = current_pts
            best_global_beta = current_beta
            best_global_center = current_center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
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

    # Final safety scale
    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
