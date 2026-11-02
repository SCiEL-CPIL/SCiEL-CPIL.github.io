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

    def _elapsed():
        return time.time() - start_time

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

    def _polish(pos, rad, n_iters=100):
        """Iteratively recompute max radii for given positions."""
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

    def _high_precision_polish(pos, rad, n_iters=200):
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
                new_r = max(1e-10, min(min_dist_others, min_dist_bound) - 1e-10)
                if abs(new_r - rad[i]) > 1e-14:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
        return rad

    def _local_search(pos, rad, max_evals=500):
        """
        Local search on positions using Nelder-Mead.
        Objective: minimize -sum(polished_radii).
        """
        # Initial radii from polish
        rad_init = _polish(pos.copy(), rad.copy(), n_iters=50)
        
        def pos_objective(p):
            # p is 2N vector of positions
            trial_pos = np.column_stack((p[0::2], p[1::2]))
            # Clamp to [0,1] to help optimizer
            trial_pos[:, 0] = np.clip(trial_pos[:, 0], 0, 1)
            trial_pos[:, 1] = np.clip(trial_pos[:, 1], 0, 1)
            
            # Calculate max radii for this configuration
            trial_rad = _polish(trial_pos, np.zeros(N), n_iters=20)
            
            # Check validity
            score = _evaluate(trial_pos, trial_rad)
            if score == -np.inf:
                return 1e6 # Large penalty for invalid
            return -score
        
        p0 = np.zeros(2 * N)
        p0[0::2] = pos[:, 0]
        p0[1::2] = pos[:, 1]
        
        try:
            res = minimize(pos_objective, p0, method='Nelder-Mead', 
                           options={'maxiter': max_evals, 'xatol': 1e-10, 'fatol': 1e-10})
            p_opt = res.x
            pos_new = np.column_stack((p_opt[0::2], p_opt[1::2]))
            pos_new[:, 0] = np.clip(pos_new[:, 0], 0, 1)
            pos_new[:, 1] = np.clip(pos_new[:, 1], 0, 1)
            
            # Final polish on the best found position
            rad_new = _polish(pos_new, rad.copy(), n_iters=50)
            return pos_new, rad_new
        except Exception:
            return pos.copy(), rad.copy()

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(10): # Reduced random inits to allow more time for local search
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if _elapsed() > deadline - 10:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # 1. Coarse Optimization with L-BFGS-B
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                           jac=lambda v: _objective_grad(v, 500.0),
                           bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxls': 50})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # 2. Polish Radii
        R = _polish(pos, R, n_iters=50)
        
        # 3. Local Search on Positions (Nelder-Mead)
        if _elapsed() < deadline - 15:
            pos, R = _local_search(pos, R, max_evals=200)
        
        score = _evaluate(pos, R)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # Final high-precision polish on the best result
    if best_result is not None:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        
        # One more local search pass if time permits
        if _elapsed() < deadline - 5:
            pos, rad = _local_search(pos, rad, max_evals=100)
            
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
