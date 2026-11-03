import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 3.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate candidate configurations
    candidates = []

    # Standard hexagonal cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Dense 4-wide
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    # New compact configurations
    # Compact diamond
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(0,-2),(2,-1)])
    # Hex ring + center + 4
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Compact L-shape
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,-1),(2,1)])
    # Dense 3x4
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Asymmetric compact
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(-1,1),(3,1),(0,-1)])
    # Tight cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)])
    # Offset rows
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Compact V
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(0,1),(-1,0)])
    # Dense asymmetric
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Ultra compact
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,2)])
    # Additional compact shapes
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(2,3),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(1,3),(2,3),(0,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(0,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(3,0)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,1),(2,-1),(-2,1)])

    def calc_S_for_beta(pts, beta):
        """Vectorized LP for minimum enclosing hexagon with given rotation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        # For each normal and point: two constraints
        # Total: 12 * n_pts
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        # Vectorize: for each normal k, project all points
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny  # (n_pts,)
            start = 2 * k * n_pts
            # Constraint 1: -nx*cx - ny*cy - D <= -proj_i
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            # Constraint 2: nx*cx + ny*cy - D <= proj_i
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
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def hexagon_vertices(cx, cy, angle_deg):
        """Return 6 vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = a + k * math.pi / 3.0
            verts.append((cx + math.cos(ang), cy + math.sin(ang)))
        return verts

    def poly_overlap(v1, v2, tol=1e-7):
        """Check if two convex polygons overlap (interior intersection)."""
        # Quick bbox reject
        xs1 = [v[0] for v in v1]; ys1 = [v[1] for v in v1]
        xs2 = [v[0] for v in v2]; ys2 = [v[1] for v in v2]
        if max(xs1) < min(xs2) + tol or max(xs2) < min(xs1) + tol:
            return False
        if max(ys1) < min(ys2) + tol or max(ys2) < min(ys1) + tol:
            return False
        # Separating axis theorem
        for poly in (v1, v2):
            n = len(poly)
            for i in range(n):
                ax1, ay1 = poly[i]
                ax2, ay2 = poly[(i+1) % n]
                # Edge normal
                dx = ax2 - ax1
                dy = ay2 - ay1
                # Normal perpendicular to edge
                nx, ny = -dy, dx
                norm = math.sqrt(nx*nx + ny*ny)
                if norm < 1e-12:
                    continue
                nx /= norm; ny /= norm
                proj1 = [v[0]*nx + v[1]*ny for v in v1]
                proj2 = [v[0]*nx + v[1]*ny for v in v2]
                if max(proj1) < min(proj2) + tol or max(proj2) < min(proj1) + tol:
                    return False
        return True

    def check_all_disjoint(pts, angles_deg, tol=1e-7):
        """Check all pairs of hexagons are disjoint."""
        n = len(pts)
        verts_list = []
        for i in range(n):
            verts_list.append(hexagon_vertices(pts[i,0], pts[i,1], angles_deg[i]))
        for i in range(n):
            for j in range(i+1, n):
                if poly_overlap(verts_list[i], verts_list[j], tol):
                    return False
        return True

    def continuous_relax(pts, beta, center, max_iters=100):
        """Efficient position relaxation using subgradient descent on the smooth surrogate.
        Returns improved (pts, beta, center, S) or None."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        n_pts = len(pts)
        cur_pts = pts.copy()
        cur_beta = beta
        cur_center = center
        cur_S = exact_S(cur_pts, cur_center[0], cur_center[1], cur_beta)
        
        angles_deg = [0.0] * n_pts
        
        # Use subgradient descent on the smooth surrogate
        temperature = 0.05
        lr = 0.02
        best_S = cur_S
        best_pts = cur_pts.copy()
        best_beta = cur_beta
        best_center = cur_center
        
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute subgradients for each hexagon position
            angles = cur_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Centered projections
            pts_centered = cur_pts - np.array([cur_center[0], cur_center[1]])
            projs = pts_centered @ normals.T  # (n_pts, 6)
            
            # Find which constraint is most active for each hexagon
            # Subgradient of D w.r.t. point i position
            grads = np.zeros_like(cur_pts)
            for k in range(6):
                nx, ny = normals[k]
                # For each point, check if it's the max in + or - direction
                for i in range(n_pts):
                    val_pos = projs[i, k]
                    val_neg = -projs[i, k]
                    max_pos = np.max(projs[:, k])
                    max_neg = np.max(-projs[:, k])
                    
                    if abs(val_pos - max_pos) < 1e-9:
                        grads[i, 0] += nx / n_pts
                        grads[i, 1] += ny / n_pts
                    if abs(val_neg - max_neg) < 1e-9:
                        grads[i, 0] -= nx / n_pts
                        grads[i, 1] -= ny / n_pts
            
            # Update positions
            for i in range(n_pts):
                cur_pts[i, 0] -= lr * grads[i, 0]
                cur_pts[i, 1] -= lr * grads[i, 1]
            
            # Check disjointness
            if not check_all_disjoint(cur_pts, angles_deg):
                # Revert
                cur_pts = cur_pts + lr * grads
                continue
            
            # Recompute S
            S_test = exact_S(cur_pts, cur_center[0], cur_center[1], cur_beta)
            
            # Try to improve center and beta
            S_test2, c_test, _ = calc_S_for_beta(cur_pts, cur_beta)
            if S_test2 < S_test:
                S_test = S_test2
                cur_center = c_test
            
            if S_test < best_S - 1e-10:
                best_S = S_test
                best_pts = cur_pts.copy()
                best_beta = cur_beta
                best_center = cur_center
                # Decay learning rate
                lr *= 0.95
            else:
                lr *= 0.8
        
        if best_S < cur_S - 1e-10:
            return best_pts, best_beta, best_center, best_S
        return None

    def optimize_shape(pts):
        """Multi-stage optimization for a fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Stage 1: Finer grid search over beta in [0, pi/6]
        n_grid = 73
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

        # Stage 2: Scalar refinement of beta with tighter tolerance
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Stage 3: Joint (cx, cy, beta) optimization with smooth surrogate
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02, 0.0, math.pi / 12, 
                                   best_beta + 0.04, best_beta - 0.04, best_beta + 0.01, best_beta - 0.01]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 1200, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        # Stage 3.5: Continuous relaxation - move hexagons off lattice
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                relax_result = continuous_relax(pts, best_beta, best_center, max_iters=80)
                if relax_result is not None:
                    rp, rb, rc, rS = relax_result
                    if rS < best_S:
                        best_S = rS
                        best_beta = rb
                        best_center = rc
                        pts = rp  # Update for subsequent stages
        except Exception:
            pass

        # Stage 4: High-precision polish with exact objective - more starts, tighter tol
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                delta = 0.005
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + delta],
                    [best_center[0], best_center[1], best_beta - delta],
                    [best_center[0] + delta, best_center[1], best_beta],
                    [best_center[0] - delta, best_center[1], best_beta],
                    [best_center[0], best_center[1] + delta, best_beta],
                    [best_center[0], best_center[1] - delta, best_beta],
                    [best_center[0] + delta*0.5, best_center[1] + delta*0.5, best_beta],
                    [best_center[0] - delta*0.5, best_center[1] - delta*0.5, best_beta],
                    [best_center[0] + delta*0.5, best_center[1] - delta*0.5, best_beta],
                    [best_center[0] - delta*0.5, best_center[1] + delta*0.5, best_beta],
                    [best_center[0], best_center[1], best_beta + delta*0.5],
                    [best_center[0], best_center[1], best_beta - delta*0.5],
                    [best_center[0] + delta*0.3, best_center[1] - delta*0.3, best_beta + delta*0.2],
                    [best_center[0] - delta*0.3, best_center[1] + delta*0.3, best_beta - delta*0.2],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Stage 5: Final LP-based center refinement at fixed optimal beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                S_final, center_final, D_final = calc_S_for_beta(pts, best_beta)
                if S_final < best_S:
                    best_S = S_final
                    best_center = center_final
                
                # Additional: try a few nearby betas with LP
                for dbeta in [0.001, -0.001, 0.002, -0.002, 0.0005, -0.0005]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    test_beta = ((best_beta + dbeta) % (math.pi / 6) + math.pi / 6) % (math.pi / 6)
                    S_test, c_test, _ = calc_S_for_beta(pts, test_beta)
                    if S_test < best_S:
                        best_S = S_test
                        best_beta = test_beta
                        best_center = c_test
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick heuristic: bounding box size
    def quick_bbox_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        return np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1])

    sorted_candidates = sorted(candidates, key=quick_bbox_score)

    # First pass: evaluate top candidates with full optimization
    # Evaluate top 25 candidates to ensure good coverage
    for idx, axial in enumerate(sorted_candidates[:25]):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
    
    # Second pass: if time remains, try remaining candidates with lighter optimization
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 10:
        for idx, axial in enumerate(sorted_candidates[25:]):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts = axial_to_cartesian(axial)
            # Quick evaluation: just grid search over beta
            best_S_quick = float('inf')
            best_beta_quick = 0.0
            best_center_quick = None
            for i in range(37):
                beta = i * (math.pi / 6) / 36
                S, center, D = calc_S_for_beta(pts, beta)
                if S < best_S_quick:
                    best_S_quick = S
                    best_beta_quick = beta
                    best_center_quick = center
            if best_center_quick is not None and best_S_quick < best_global_S:
                best_global_S = best_S_quick
                best_global_pts = pts
                best_global_beta = best_beta_quick
                best_global_center = best_center_quick

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

    # Final: compute side length with tiny safety scale
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

    # Verify disjointness of final configuration
    angles_deg_final = [angle_deg] * len(pts_scaled)
    if not check_all_disjoint(pts_scaled, angles_deg_final, tol=1e-6):
        # Fallback: use original lattice positions (which are guaranteed disjoint)
        pts_final = best_global_pts.copy()
        max_proj_orig = 0
        for k in range(6):
            ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
            max_proj_orig = max(max_proj_orig, np.max(projs), -np.min(projs))
        S_scaled = (max_proj_orig + APOTHEM) / APOTHEM
    else:
        pts_final = pts_scaled

    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
