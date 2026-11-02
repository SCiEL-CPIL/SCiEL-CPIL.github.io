"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    # --- Helper Functions ---

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

    def _polish(pos, rad, n_iters=20):
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
                new_r = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-6)
                if abs(new_r - rad[i]) > 1e-10:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
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

    def _local_search_anneal(pos, rad, time_budget=2.0):
        """
        Structural Change: Simulated Annealing style local search.
        Perturbs positions, recalculates radii, and accepts moves based on score improvement
        or probabilistic acceptance to escape local minima.
        """
        current_pos = pos.copy()
        current_rad = rad.copy()
        current_score = _evaluate(current_pos, current_rad)
        if np.isneginf(current_score):
             return current_pos, current_rad, current_score

        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_score = current_score

        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        
        # Cooling schedule
        T = 0.05
        T_min = 0.001
        cooling_rate = 0.95
        max_steps = 2000
        
        steps_done = 0
        start_ls = time.time()
        
        while steps_done < max_steps and (time.time() - start_ls) < time_budget:
            # Perturb a random subset of circles
            n_perturb = rng.randint(1, 4)
            indices = rng.choice(N, n_perturb, replace=False)
            
            new_pos = current_pos.copy()
            # Gaussian perturbation
            noise = rng.normal(0, 0.02, size=(n_perturb, 2))
            new_pos[indices] += noise
            
            # Keep inside bounds
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0.05, 0.95)
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0.05, 0.95)
            
            # Recalculate radii for new positions
            new_rad = _polish(new_pos, current_rad.copy(), n_iters=5)
            
            new_score = _evaluate(new_pos, new_rad)
            
            if np.isneginf(new_score):
                # Invalid move, reject
                T *= cooling_rate
                steps_done += 1
                continue
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta > 0 or rng.rand() < np.exp(delta / T):
                current_pos = new_pos
                current_rad = new_rad
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pos = current_pos.copy()
                    best_rad = current_rad.copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            steps_done += 1
            
        return best_pos, best_rad, best_score

    # --- Main Execution Loop ---

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations: hex, grid, and several random
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(15): # Reduced random seeds to allow time for local search
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 10:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # Phase 1: Global Optimization (L-BFGS-B)
        # Use fewer iterations for random inits to save time for local search
        maxiter = 200 if idx < 2 else 100
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Initial Polish
        R = _polish(pos, R, n_iters=10)
        score = _evaluate(pos, R)
        
        # Phase 2: Local Search (Annealing) - The Structural Improvement
        # Only run local search if we have time and the current solution is valid
        if score > -np.inf and time.time() < deadline - 5:
            # Allocate time for local search: more for promising starts (hex/grid)
            ls_time = 3.0 if idx < 2 else 1.5
            pos_ls, R_ls, score_ls = _local_search_anneal(pos, R, time_budget=ls_time)
            
            if score_ls > score:
                pos = pos_ls
                R = R_ls
                score = score_ls

        # Final Polish on the locally searched result
        R = _polish(pos, R, n_iters=20)
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
    if time.time() < deadline - 2:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        
        # One more tight L-BFGS-B pass with tight bounds
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = pos[i, 0]
            x0[3 * i + 1] = pos[i, 1]
            x0[3 * i + 2] = rad[i]
        
        tight_bounds = []
        for i in range(N):
            tight_bounds.append((max(0, pos[i,0] - 0.05), min(1, pos[i,0] + 0.05)))
            tight_bounds.append((max(0, pos[i,1] - 0.05), min(1, pos[i,1] + 0.05)))
            tight_bounds.append((max(1e-5, rad[i] * 0.8), min(0.5, rad[i] * 1.2)))
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=tight_bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x
            X = x_opt[0::3].copy()
            Y = x_opt[1::3].copy()
            R = x_opt[2::3].copy()
            pos_p = np.column_stack((X, Y))
            R = _polish(pos_p, R, n_iters=30)
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
