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

    # Define candidate point sets (axial coordinates)
    candidates = []

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
        (0, 0), (1, 0),
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

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3), (1, 3), (-1, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 2), (3, 2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 2), (-1, 2), (1, -1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_32)

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

    def calc_S_exact(pts, cx, cy, beta):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_val = np.max(np.abs(projs))
        S = (max_val + APOTHEM) / APOTHEM
        return S, max_val

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def calc_S_smooth_with_grad(pts, cx, cy, beta, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), np.zeros(3)
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum = np.log(sum_exp) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        
        probs = exp_shifted / sum_exp
        n_pts = len(pts)
        
        probs_pos = probs[:n_pts * 6].reshape(n_pts, 6)
        probs_neg = probs[n_pts * 6:].reshape(n_pts, 6)
        
        diff_probs = probs_neg - probs_pos
        
        grad_cx = np.sum(diff_probs * normals[:, 0])
        grad_cy = np.sum(diff_probs * normals[:, 1])
        
        v_x = pts_centered[:, 0:1]
        v_y = pts_centered[:, 1:2]
        sin_a = normals[:, 1:2].T
        cos_a = normals[:, 0:1].T
        
        dp_da = -v_x * sin_a + v_y * cos_a
        
        grad_beta = -np.sum(diff_probs * dp_da)
        
        grad_D = np.array([grad_cx, grad_cy, grad_beta])
        grad_S = grad_D / APOTHEM
        return S, grad_S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12, math.pi/6, best_beta + 0.1, best_beta - 0.1]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        def smooth_obj_grad(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, grad = calc_S_smooth_with_grad(pts, cx, cy, beta, temperature=t)
                            if S > 1e10:
                                return S, np.zeros(3)
                            return S, grad
                        
                        init = [best_center[0], best_center[1], sb]
                        
                        try:
                            res = minimize(smooth_obj_grad, init, method='L-BFGS-B', jac=True,
                                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                                S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = c_exact
                        except Exception:
                            pass
                        
                        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                                S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = c_exact
        except Exception:
            pass
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    S, _ = calc_S_exact(pts, cx, cy, beta)
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                    [best_center[0], best_center[1], best_beta - 0.01],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0] - 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.01, best_beta],
                    [best_center[0] + 0.005, best_center[1] + 0.005, best_beta],
                    [best_center[0] - 0.005, best_center[1] - 0.005, best_beta],
                    [best_center[0] + 0.005, best_center[1] - 0.005, best_beta],
                    [best_center[0] - 0.005, best_center[1] + 0.005, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                        res = minimize(objective, init, method='Powell',
                                     options={'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-14})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    # Continuous NLP refinement: optimize positions and orientations directly
    def hex_vertices(cx, cy, angle):
        """Return 6 vertices of a unit hexagon centered at (cx,cy) with given rotation."""
        verts = []
        for k in range(6):
            ang = angle + k * math.pi / 3.0
            verts.append((cx + math.cos(ang), cy + math.sin(ang)))
        return np.array(verts)

    def min_distance_between_hexagons(pts_i, ang_i, pts_j, ang_j):
        """
        Compute the minimum distance between two hexagons.
        Uses the fact that for two convex polygons, the minimum distance
        can be found by checking all vertex-edge pairs.
        Returns (min_dist, grad_i, grad_j) where grads are w.r.t. centers.
        """
        verts_i = hex_vertices(pts_i[0], pts_i[1], ang_i)
        verts_j = hex_vertices(pts_j[0], pts_j[1], ang_j)
        
        min_dist = float('inf')
        best_pair = None
        
        for vi in verts_i:
            for vj in verts_j:
                d = np.linalg.norm(vi - vj)
                if d < min_dist:
                    min_dist = d
                    best_pair = (vi, vj)
        
        # Also check vertex-to-edge distances
        for vi in verts_i:
            for k in range(6):
                vj1 = verts_j[k]
                vj2 = verts_j[(k+1) % 6]
                edge_vec = vj2 - vj1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-15:
                    continue
                t = np.dot(vi - vj1, edge_vec) / (edge_len ** 2)
                t = max(0, min(1, t))
                closest = vj1 + t * edge_vec
                d = np.linalg.norm(vi - closest)
                if d < min_dist:
                    min_dist = d
                    best_pair = (vi, closest)
        
        for vj in verts_j:
            for k in range(6):
                vi1 = verts_i[k]
                vi2 = verts_i[(k+1) % 6]
                edge_vec = vi2 - vi1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-15:
                    continue
                t = np.dot(vj - vi1, edge_vec) / (edge_len ** 2)
                t = max(0, min(1, t))
                closest = vi1 + t * edge_vec
                d = np.linalg.norm(vj - closest)
                if d < min_dist:
                    min_dist = d
                    best_pair = (closest, vj)
        
        # Compute gradients numerically for the smooth surrogate
        # We'll use a soft-min over all pair distances
        return min_dist

    def smooth_packing_objective(params, pts_init, beta_init, temperature=0.05):
        """
        Objective for continuous packing optimization.
        params: [x_0, y_0, a_0, x_1, y_1, a_1, ..., x_10, y_10, a_10, cx, cy, beta]
        Minimize the outer hexagon side length while maintaining non-overlap.
        """
        n = N
        # Extract positions and angles
        positions = []
        angles = []
        for i in range(n):
            positions.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        cx, cy, beta = params[3*n], params[3*n+1], params[3*n+2]
        
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        # Compute outer hexagon constraint
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_arr = np.array(positions)
        pts_centered = pts_arr - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return 1e10
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        
        # Non-overlap penalty: for each pair, ensure distance >= 2*APOTHEM
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                vi = hex_vertices(positions[i][0], positions[i][1], angles[i])
                vj = hex_vertices(positions[j][0], positions[j][1], angles[j])
                
                # Compute minimum distance between the two hexagons
                min_d = float('inf')
                for a in vi:
                    for b in vj:
                        d = np.linalg.norm(a - b)
                        if d < min_d:
                            min_d = d
                for a in vi:
                    for k in range(6):
                        b1 = vj[k]
                        b2 = vj[(k+1)%6]
                        ev = b2 - b1
                        el = np.linalg.norm(ev)
                        if el < 1e-15: continue
                        t = np.dot(a - b1, ev) / (el**2)
                        t = max(0, min(1, t))
                        c = b1 + t * ev
                        d = np.linalg.norm(a - c)
                        if d < min_d: min_d = d
                for b in vj:
                    for k in range(6):
                        a1 = vi[k]
                        a2 = vi[(k+1)%6]
                        ev = a2 - a1
                        el = np.linalg.norm(ev)
                        if el < 1e-15: continue
                        t = np.dot(b - a1, ev) / (el**2)
                        t = max(0, min(1, t))
                        c = a1 + t * ev
                        d = np.linalg.norm(b - c)
                        if d < min_d: min_d = d
                
                required = 2.0 * APOTHEM
                if min_d < required:
                    deficit = required - min_d
                    penalty += deficit ** 2
        
        return S + 100.0 * penalty

    def continuous_refine(pts_init, beta_init, cx_init, cy_init):
        """Perform continuous NLP refinement starting from axial lattice configuration."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None
        
        n = N
        # Initial parameters: positions from axial lattice, angles = 0
        params = []
        for x, y in pts_init:
            params.extend([float(x), float(y), 0.0])
        params.extend([float(cx_init), float(cy_init), float(beta_init)])
        
        best_S = float('inf')
        best_params = None
        
        # Try with different temperatures and methods
        temperatures = [0.1, 0.05, 0.02]
        
        for temp in temperatures:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            def obj(p, t=temp):
                return smooth_packing_objective(p, pts_init, beta_init, temperature=t)
            
            try:
                res = minimize(obj, params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            try:
                res = minimize(obj, params, method='Powell',
                             options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        if best_params is None:
            return float('inf'), None, None, None
        
        # Extract final configuration
        final_positions = []
        final_angles = []
        for i in range(n):
            final_positions.append((best_params[3*i], best_params[3*i+1]))
            final_angles.append(best_params[3*i+2])
        final_cx, final_cy, final_beta = best_params[3*n], best_params[3*n+1], best_params[3*n+2]
        final_beta = ((final_beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        # Verify non-overlap
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                vi = hex_vertices(final_positions[i][0], final_positions[i][1], final_angles[i])
                vj = hex_vertices(final_positions[j][0], final_positions[j][1], final_angles[j])
                min_d = float('inf')
                for a in vi:
                    for b in vj:
                        d = np.linalg.norm(a - b)
                        if d < min_d: min_d = d
                for a in vi:
                    for k in range(6):
                        b1 = vj[k]
                        b2 = vj[(k+1)%6]
                        ev = b2 - b1
                        el = np.linalg.norm(ev)
                        if el < 1e-15: continue
                        t = np.dot(a - b1, ev) / (el**2)
                        t = max(0, min(1, t))
                        c = b1 + t * ev
                        d = np.linalg.norm(a - c)
                        if d < min_d: min_d = d
                for b in vj:
                    for k in range(6):
                        a1 = vi[k]
                        a2 = vi[(k+1)%6]
                        ev = a2 - a1
                        el = np.linalg.norm(ev)
                        if el < 1e-15: continue
                        t = np.dot(b - a1, ev) / (el**2)
                        t = max(0, min(1, t))
                        c = a1 + t * ev
                        d = np.linalg.norm(b - c)
                        if d < min_d: min_d = d
                if min_d < 2.0 * APOTHEM - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return float('inf'), None, None, None
        
        # Compute exact S for this configuration
        pts_arr = np.array(final_positions)
        S, _ = calc_S_exact(pts_arr, final_cx, final_cy, final_beta)
        
        return S, final_positions, final_angles, (final_cx, final_cy, final_beta)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_positions = None
    best_global_angles = None

    # First pass: quick evaluation to find top candidates
    quick_results = []
    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        best_quick_S = float('inf')
        best_quick_beta = 0.0
        best_quick_center = None
        for j in range(13):
            beta = j * (math.pi / 6) / 12
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_quick_S:
                best_quick_S = S
                best_quick_beta = beta
                best_quick_center = center
        quick_results.append((best_quick_S, i, pts, best_quick_beta, best_quick_center))
    
    quick_results.sort(key=lambda x: x[0])
    
    # Phase 1: Lattice-based optimization (top 8 candidates)
    max_detailed = min(len(quick_results), 8)
    for idx in range(max_detailed):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

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

    # Phase 2: Continuous NLP refinement on top candidates
    max_continuous = min(len(quick_results), 6)
    for idx in range(max_continuous):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        quick_S, axial_idx, pts, quick_beta, quick_center = quick_results[idx]
        
        S_cont, positions, angles, center_beta = continuous_refine(pts, quick_beta, quick_center[0], quick_center[1])
        if S_cont < best_global_S:
            best_global_S = S_cont
            best_global_pts = np.array(positions)
            best_global_beta = center_beta[2]
            best_global_center = (center_beta[0], center_beta[1])
            best_global_positions = positions
            best_global_angles = angles

    # Final output
    if best_global_positions is not None and best_global_angles is not None:
        cx, cy = best_global_center
        angle_deg = math.degrees(best_global_beta)
        
        inner = []
        for i in range(N):
            x, y = best_global_positions[i]
            ang = best_global_angles[i]
            inner.append([float(x), float(y), float(math.degrees(ang))])
        
        center = [float(cx), float(cy)]
        
        # Compute exact side length
        pts_arr = np.array(best_global_positions)
        S_exact, _ = calc_S_exact(pts_arr, cx, cy, best_global_beta)
        side = float(S_exact)
        rotation = float(angle_deg)
        
        return inner, center, side, rotation
    else:
        cx, cy = best_global_center
        angle_deg = math.degrees(best_global_beta)
        
        SCALE = 1.0 + 1e-12
        pts_scaled = []
        for x, y in best_global_pts:
            sx = cx + (x - cx) * SCALE
            sy = cy + (y - cy) * SCALE
            pts_scaled.append((sx, sy))
        
        pts_scaled = np.array(pts_scaled)
        
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
