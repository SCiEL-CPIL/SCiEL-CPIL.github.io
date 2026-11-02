"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    best_score = -np.inf
    best_result = None

    def _make_hex_init():
        """Create hexagonal lattice initialization."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        return init_pts

    def _make_grid_init():
        """Create grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([0.5/side + i/side, 0.5/side + j/side])
        return np.array(centers[:N])

    def _make_random_init(seed):
        """Create random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def _compute_initial_radius(pts):
        """Compute initial radius based on minimum pairwise distance."""
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2 * 0.85

    def _objective(vars_, penalty_weight=500.0):
        """Objective: minimize -sum(radii) + penalty for violations."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    pen += overlap ** 2
            if X[i] - R[i] < 0:
                pen += (X[i] - R[i]) ** 2
            if 1.0 - X[i] - R[i] < 0:
                pen += (1.0 - X[i] - R[i]) ** 2
            if Y[i] - R[i] < 0:
                pen += (Y[i] - R[i]) ** 2
            if 1.0 - Y[i] - R[i] < 0:
                pen += (1.0 - Y[i] - R[i]) ** 2
        return obj + penalty_weight * pen

    def _objective_grad(vars_, penalty_weight=500.0):
        """Analytical gradient of the objective."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        weight = penalty_weight
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    continue
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    factor = 2.0 * overlap
                    grad[3 * i] += weight * factor * (-dx / d)
                    grad[3 * i + 1] += weight * factor * (-dy / d)
                    grad[3 * j] += weight * factor * (dx / d)
                    grad[3 * j + 1] += weight * factor * (dy / d)
                    grad[3 * i + 2] += weight * factor
                    grad[3 * j + 2] += weight * factor
            val = X[i] - R[i]
            if val < 0:
                grad[3 * i] += weight * 2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = 1.0 - X[i] - R[i]
            if val < 0:
                grad[3 * i] += weight * -2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = Y[i] - R[i]
            if val < 0:
                grad[3 * i + 1] += weight * 2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = 1.0 - Y[i] - R[i]
            if val < 0:
                grad[3 * i + 1] += weight * -2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
        return grad

    def _objective_smooth(vars_, penalty_weight=1000.0, beta=20.0):
        """Smooth objective using log-sum-exp for overlap penalty."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        # Collect all constraint violations
        violations = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                violations.append(R[i] + R[j] - d)
            violations.append(X[i] - R[i])
            violations.append(1.0 - X[i] - R[i])
            violations.append(Y[i] - R[i])
            violations.append(1.0 - Y[i] - R[i])
        v = np.array(violations)
        # Soft penalty: sum of soft-max(v, 0)^2 approximated
        # Use: soft_max(v, 0) = log(1 + exp(beta*v)) / beta for v > 0
        # But we want penalty only when v > 0, so: max(v,0)^2 ~ (soft_max(v,0))^2
        # Simpler: use smooth hinge: (v + sqrt(v^2 + eps))/2 for v, then square
        eps = 1e-4
        soft_v = (v + np.sqrt(v * v + eps)) / 2.0
        pen = np.sum(soft_v ** 2)
        return obj + penalty_weight * pen

    def _objective_smooth_grad(vars_, penalty_weight=1000.0, beta=20.0):
        """Gradient of smooth objective."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        eps = 1e-4
        weight = penalty_weight
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    continue
                v = R[i] + R[j] - d
                # d/dv of soft_v^2 where soft_v = (v + sqrt(v^2+eps))/2
                # d(soft_v)/dv = (1 + v/sqrt(v^2+eps))/2
                sv = (v + np.sqrt(v * v + eps)) / 2.0
                dsv_dv = (1.0 + v / np.sqrt(v * v + eps)) / 2.0
                # d(v)/d(X[i]) = -dx/d, d(v)/d(X[j]) = dx/d
                # d(v)/d(R[i]) = 1, d(v)/d(R[j]) = 1
                factor = weight * 2.0 * sv * dsv_dv
                grad[3 * i] += factor * (-dx / d)
                grad[3 * i + 1] += factor * (-dy / d)
                grad[3 * j] += factor * (dx / d)
                grad[3 * j + 1] += factor * (dy / d)
                grad[3 * i + 2] += factor
                grad[3 * j + 2] += factor
        
        # Boundary constraints
        for i in range(N):
            for idx, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1)]):
                if axis == 0:
                    v = X[i] - R[i] if sign > 0 else 1.0 - X[i] - R[i]
                else:
                    v = Y[i] - R[i] if sign > 0 else 1.0 - Y[i] - R[i]
                sv = (v + np.sqrt(v * v + eps)) / 2.0
                dsv_dv = (1.0 + v / np.sqrt(v * v + eps)) / 2.0
                factor = weight * 2.0 * sv * dsv_dv
                pos_idx = 3 * i + axis
                if sign > 0:
                    grad[pos_idx] += factor
                    grad[3 * i + 2] += -factor
                else:
                    grad[pos_idx] += -factor
                    grad[3 * i + 2] += -factor
        return grad

    def _evaluate(centers, radii):
        """Compute sum of radii if valid, else -inf."""
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return -np.inf
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return -np.inf
        return np.sum(radii)

    def _polish(pos, rad, n_iters=30):
        """Iteratively recompute max radii for given positions with more iterations."""
        for _ in range(n_iters):
            changed = False
            for i in range(N):
                min_dist_others = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist_others = min(min_dist_others, d - rad[j])
                min_dist_bound = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                new_r = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-6)
                if abs(new_r - rad[i]) > 1e-10:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
        return rad

    def _high_precision_polish(pos, rad, n_iters=50):
        """High-precision polish with tighter tolerances."""
        for _ in range(n_iters):
            changed = False
            for i in range(N):
                min_dist_others = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist_others = min(min_dist_others, d - rad[j])
                min_dist_bound = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                new_r = max(1e-8, min(min_dist_others, min_dist_bound) - 1e-8)
                if abs(new_r - rad[i]) > 1e-12:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
        return rad

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations: hex, grid, and several random
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(30):
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 5:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # Two-phase optimization: first with quadratic penalty, then smooth
        maxiter = 400 if idx < 5 else 250
        use_smooth = idx % 2 == 0  # alternate between methods

        if use_smooth:
            try:
                res = minimize(_objective_smooth, x0, method='L-BFGS-B',
                               jac=lambda v: _objective_smooth_grad(v, 1000.0, 20.0),
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxls': 50})
                x_opt = res.x
            except Exception:
                x_opt = x0
        else:
            try:
                res = minimize(_objective, x0, method='L-BFGS-B',
                               jac=lambda v: _objective_grad(v, 500.0),
                               bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxls': 50})
                x_opt = res.x
            except Exception:
                x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Polish radii
        R = _polish(pos, R, n_iters=30)

        score = _evaluate(pos, R)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # Final high-precision polish on the best result
    if best_result is not None:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        
        # Do a local position optimization with high penalty to fine-tune
        if time.time() < deadline - 10:
            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3 * i] = pos[i, 0]
                x0[3 * i + 1] = pos[i, 1]
                x0[3 * i + 2] = rad[i]
            try:
                res = minimize(_objective, x0, method='L-BFGS-B',
                               jac=lambda v: _objective_grad(v, 2000.0),
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 50})
                x_opt = res.x
                pos2 = np.column_stack((x_opt[0::3], x_opt[1::3]))
                rad2 = x_opt[2::3].copy()
                rad2 = _polish(pos2, rad2, n_iters=50)
                score2 = _evaluate(pos2, rad2)
                if score2 > best_score:
                    best_score = score2
                    best_result = np.column_stack([pos2, rad2])
                    pos, rad = pos2, rad2
            except Exception:
                pass

        rad = _high_precision_polish(pos, rad, n_iters=100)
        final_score = _evaluate(pos, rad)
        if final_score > best_score:
            best_score = final_score
            best_result = np.column_stack([pos, rad])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
