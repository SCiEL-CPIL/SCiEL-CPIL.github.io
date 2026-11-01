"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
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

    def _objective(vars_):
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
        return obj + 100.0 * pen

    def _objective_grad(vars_):
        """Analytical gradient of the objective."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        weight = 100.0
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

    def _polish(pos, rad, n_iters=10):
        """Iteratively recompute max radii for given positions."""
        for _ in range(n_iters):
            for i in range(N):
                min_dist_others = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist_others = min(min_dist_others, d - rad[j])
                min_dist_bound = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                rad[i] = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-6)
        return rad

    def _validity_check(arr):
        """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
        if arr.shape != (N, 3):
            return False
        x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
        if np.any(r <= 0):
            return False
        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
            return False
        if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-7:
                    return False
        return True

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations: hex, grid, and several random
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(20):
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 3:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # For later restarts, use fewer iterations to save time
        maxiter = 300 if idx < 3 else 150

        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Polish radii
        R = _polish(pos, R, n_iters=5)

        score = _evaluate(pos, R)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final high-precision polish on the best result
    if time.time() < deadline - 5:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        
        # Do a tighter L-BFGS-B pass with the current best as starting point
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = pos[i, 0]
            x0[3 * i + 1] = pos[i, 1]
            x0[3 * i + 2] = rad[i]
        
        # Tighter bounds around current solution
        tight_bounds = []
        for i in range(N):
            tight_bounds.append((max(0, pos[i,0] - 0.05), min(1, pos[i,0] + 0.05)))
            tight_bounds.append((max(0, pos[i,1] - 0.05), min(1, pos[i,1] + 0.05)))
            tight_bounds.append((max(1e-5, rad[i] * 0.8), min(0.5, rad[i] * 1.2)))
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=tight_bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            x_opt = res.x
            X = x_opt[0::3].copy()
            Y = x_opt[1::3].copy()
            R = x_opt[2::3].copy()
            pos_p = np.column_stack((X, Y))
            R = _polish(pos_p, R, n_iters=10)
            score_p = _evaluate(pos_p, R)
            if score_p > best_score:
                best_result = np.column_stack([pos_p, R])
                best_score = score_p
        except Exception:
            pass
    
    # Final validity check and fix if needed
    if not _validity_check(best_result):
        # Reduce radii until valid
        for _ in range(50):
            best_result[:, 2] *= 0.95
            if _validity_check(best_result):
                break
    
    return best_result
