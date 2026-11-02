import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
CENTROID = np.array([0.5, H / 3.0])

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # Precompute indices for combinations
    indices = list(itertools.combinations(range(N), 3))
    n_combos = len(indices)
    
    def check_validity(points):
        x = points[:, 0]
        y = points[:, 1]
        if np.any(y < -1e-9):
            return False
        if np.any(y > SQRT3 * x + 1e-9):
            return False
        if np.any(y > -SQRT3 * x + SQRT3 + 1e-9):
            return False
        return True

    def min_area(points):
        x = points[:, 0]
        y = points[:, 1]
        min_area = float('inf')
        for i, j, k in indices:
            area = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
            if area < min_area:
                min_area = area
        return min_area

    def _random_valid_points(n):
        points = []
        for _ in range(n):
            while True:
                u, v = np.random.rand(2)
                if u + v < 1:
                    x = v * 1.0 + (1 - u - v) * 0.5
                    y = (1 - u - v) * H
                    points.append([x, y])
                    break
        return np.array(points)

    # ===== Symmetry-based parameterization =====
    # The equilateral triangle has 3-fold rotational symmetry around its centroid.
    CX, CY = 0.5, H / 3.0

    def symmetry_points(params):
        """
        params: array of shape (3, 3) where each row is [r, theta, w]
        r: radial distance from centroid
        theta: base angle
        w: weight/breaking factor (0 = fully symmetric, 1 = max breaking)
        
        For each of the 3 symmetry copies, we place points.
        Point 0 is at centroid.
        Points 1-3: first ring (3 points, 120 deg apart)
        Points 4-6: second ring
        Points 7-9: third ring
        Point 10: at centroid (or offset)
        """
        pts = np.zeros((N, 2))
        pts[0] = [CX, CY]  # centroid
        
        for ring_idx in range(3):
            r = params[ring_idx, 0]
            theta_base = params[ring_idx, 1]
            w = params[ring_idx, 2]
            
            for k in range(3):
                angle = theta_base + 2 * np.pi * k / 3.0 + w * 0.3 * np.sin(k * 2 * np.pi / 3.0)
                idx = 1 + ring_idx * 3 + k
                pts[idx, 0] = CX + r * np.cos(angle)
                pts[idx, 1] = CY + r * np.sin(angle)
        
        # Point 10: an extra point
        r10 = params[3, 0]
        theta10 = params[3, 1]
        pts[10, 0] = CX + r10 * np.cos(theta10)
        pts[10, 1] = CY + r10 * np.sin(theta10)
        
        return pts

    def project_to_triangle(points):
        """Project points to be inside the triangle (hard constraint enforcement)."""
        pts = points.copy()
        x = pts[:, 0]
        y = pts[:, 1]
        
        for i in range(N):
            px, py = pts[i]
            if py < 0:
                py = 0.0
            if py > SQRT3 * px:
                d = (SQRT3 * px - py) / 4.0
                px = px - d * SQRT3
                py = py + d
            if py > -SQRT3 * px + SQRT3:
                d = (SQRT3 * px + py - SQRT3) / 4.0
                px = px - d * SQRT3
                py = py - d
            
            px = np.clip(px, -0.1, 1.1)
            py = np.clip(py, -0.1, H + 0.1)
            pts[i] = [px, py]
        
        return pts

    def objective_sym(params):
        pts = symmetry_points(params)
        pts = project_to_triangle(pts)
        
        if not check_validity(pts):
            return 1e10
        
        ma = min_area(pts)
        if ma < 1e-12:
            return 1e10
        return -ma

    bounds_sym = [
        (0.0, H), (0.0, 2 * np.pi), (-1.0, 1.0),
        (0.0, H), (0.0, 2 * np.pi), (-1.0, 1.0),
        (0.0, H), (0.0, 2 * np.pi), (-1.0, 1.0),
        (0.0, H), (0.0, 2 * np.pi), (0.0, 0.0),
    ]

    init_sym = np.array([
        [H / 6, 0.0, 0.0],
        [H / 3, np.pi / 6, 0.0],
        [H / 2, 0.0, 0.0],
        [H / 4, np.pi / 3, 0.0],
    ])

    # ===== Phase 1: Optimize symmetry parameters =====
    elapsed = time.time() - start_time
    if elapsed < timeout - 60:
        try:
            result_sym = differential_evolution(
                objective_sym,
                bounds_sym,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                seed=42,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
            )
            best_params_sym = result_sym.x
            best_pts_sym = symmetry_points(best_params_sym)
            best_pts_sym = project_to_triangle(best_pts_sym)
            if not check_validity(best_pts_sym):
                best_pts_sym = _random_valid_points(N)
            best_score = min_area(best_pts_sym)
        except Exception:
            best_pts_sym = _random_valid_points(N)
            best_score = min_area(best_pts_sym)
    else:
        best_pts_sym = _random_valid_points(N)
        best_score = min_area(best_pts_sym)

    best_points = best_pts_sym.copy()

    # ===== Phase 2: SLSQP with auxiliary variable t =====
    elapsed = time.time() - start_time
    if elapsed > timeout - 35:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    def _area_matrix(points):
        """Compute area for all C(11,3) = 165 triangles."""
        x = points[:, 0]
        y = points[:, 1]
        areas = np.zeros(n_combos)
        for idx, (i, j, k) in enumerate(indices):
            areas[idx] = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
        return areas

    def slsqp_objective(params):
        """params: [x1,y1, x2,y2, ..., x11,y11, t]"""
        t = params[-1]
        pts = params[:N*2].reshape(N, 2)
        areas = _area_matrix(pts)
        # Penalty for constraints: areas >= t
        violation = np.maximum(0, t - areas)
        pen = np.sum(violation ** 2)
        # Penalty for triangle containment
        x = pts[:, 0]
        y = pts[:, 1]
        c1 = np.maximum(0, -y)
        c2 = np.maximum(0, y - SQRT3 * x)
        c3 = np.maximum(0, y - (-SQRT3 * x + SQRT3))
        pen += np.sum(c1**2) + np.sum(c2**2) + np.sum(c3**2)
        return -t + pen

    def slsqp_jac(params):
        t = params[-1]
        pts = params[:N*2].reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        areas = _area_matrix(pts)
        
        jac = np.zeros(N * 2 + 1)
        # d(-t)/dt = -1
        jac[-1] = -1
        
        # Gradient of penalty wrt t
        viol = np.maximum(0, t - areas)
        jac[-1] += 2 * np.sum(viol)
        
        # Gradient of penalty wrt points
        for idx, (i, j, k) in enumerate(indices):
            a = areas[idx]
            v = max(0, t - a)
            if v == 0:
                continue
            # area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
            # d(area)/d(x_i) = 0.5 * (y_j - y_k) * sign(...)
            # For simplicity, use finite differences for the gradient of the penalty
            pass
        
        # Use finite differences for the point gradients
        eps = 1e-7
        f0 = slsqp_objective(params)
        for d in range(N * 2):
            params_p = params.copy()
            params_p[d] += eps
            f_p = slsqp_objective(params_p)
            jac[d] = (f_p - f0) / eps
        
        # Add triangle containment gradient (analytic)
        for i in range(N):
            px, py = pts[i]
            # c1 = max(0, -y_i), grad wrt y_i: -1 if y_i < 0
            if py < 0:
                jac[i*2+1] += 2 * (-1) * (-1)  # 2 * c1 * d(-y)/dy = 2*0... actually 2*(-y)*(-1) = 2y < 0
                # Actually c1 = -y when y<0, so c1^2 = y^2, d/dy = 2y
                jac[i*2+1] += 2 * py
            # c2 = max(0, y_i - sqrt(3)*x_i)
            v2 = py - SQRT3 * px
            if v2 > 0:
                # c2^2 = (y - sqrt(3)x)^2, d/dx = 2(y-sqrt(3)x)(-sqrt(3)), d/dy = 2(y-sqrt(3)x)
                jac[i*2+0] += 2 * v2 * (-SQRT3)
                jac[i*2+1] += 2 * v2
            # c3 = max(0, y_i + sqrt(3)*x_i - sqrt(3))
            v3 = py + SQRT3 * px - SQRT3
            if v3 > 0:
                jac[i*2+0] += 2 * v3 * SQRT3
                jac[i*2+1] += 2 * v3
        
        return jac

    def slsqp_constraints(params):
        pts = params[:N*2].reshape(N, 2)
        t = params[-1]
        x = pts[:, 0]
        y = pts[:, 1]
        areas = _area_matrix(pts)
        # Area constraints: area_i >= t -> area_i - t >= 0
        con_areas = areas - t
        # Triangle containment: y >= 0
        con_y = y
        # y <= sqrt(3) x -> sqrt(3) x - y >= 0
        con_left = SQRT3 * x - y
        # y <= -sqrt(3) x + sqrt(3) -> sqrt(3) x + y - sqrt(3) >= 0
        con_right = SQRT3 * x + y - SQRT3
        return np.concatenate([con_areas, con_y, con_left, con_right])

    bounds_slq = [(0.0, 1.0), (0.0, H)] * N + [(0.0, 0.01)]

    try:
        res_slq = minimize(
            slsqp_objective,
            best_points.flatten().tolist() + [best_score],
            method='SLSQP',
            jac=slsqp_jac,
            bounds=bounds_slq,
            constraints=[{'type': 'ineq', 'fun': slsqp_constraints}],
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False},
        )
        if res_slq.fun < 1e9:
            new_points = res_slq.x[:N*2].reshape(N, 2)
            if check_validity(new_points):
                new_score = min_area(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > timeout - 20:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    # ===== Phase 3: Multi-start with SLSQP =====
    def objective_softmin(params, T=0.01):
        points = params.reshape(N, 2)
        x = points[:, 0]
        y = points[:, 1]
        
        pen = 0.0
        if np.any(y < 0):
            pen += np.sum(np.maximum(0, -y) ** 2)
        if np.any(y > SQRT3 * x):
            pen += np.sum(np.maximum(0, y - SQRT3 * x) ** 2)
        if np.any(y > -SQRT3 * x + SQRT3):
            pen += np.sum(np.maximum(0, y - (-SQRT3 * x + SQRT3)) ** 2)
        
        if pen > 1e-6:
            return 1e10 + pen
        
        areas = np.zeros(n_combos)
        for idx, (i, j, k) in enumerate(indices):
            areas[idx] = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
        
        areas_clipped = np.maximum(areas, 1e-12)
        log_sum_exp = np.log(np.sum(np.exp(-areas_clipped / T)))
        soft_min = -T * log_sum_exp
        
        return -soft_min

    bounds_full = [(0.0, 1.0), (0.0, H)] * N

    # Polish with soft-min
    try:
        res_lbfgs = minimize(
            objective_softmin,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds_full,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
        )
        if res_lbfgs.fun < 1e9:
            new_points = res_lbfgs.x.reshape(N, 2)
            if check_validity(new_points):
                new_score = min_area(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > timeout - 10:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    num_restarts = 2
    for i in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
        
        pts = _random_valid_points(N)
        
        try:
            res_r = minimize(
                objective_softmin,
                pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if res_r.fun < 1e9:
                if check_validity(res_r.x.reshape(N, 2)):
                    score_r = min_area(res_r.x.reshape(N, 2))
                    if score_r > best_score:
                        best_points = res_r.x.reshape(N, 2)
                        best_score = score_r
        except Exception:
            pass

    if not check_validity(best_points):
        return _random_valid_points(N)

    return best_points
