import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0
        all_points = []
        all_vals = []

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_arr = np.asarray(x, dtype=float).flatten()
            if x_arr.shape[0] != dim:
                x_arr = np.resize(x_arr, dim)
            x_clipped = np.clip(x_arr, -5, 5)
            val = float(func(x_clipped))
            eval_count += 1
            all_points.append(x_clipped.copy())
            all_vals.append(val)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Dense structured sampling to find promising basins
        # Use a combination of grid and random points for good coverage
        rng = np.random.default_rng(12345)
        
        n_grid = 4  # 4 points per dimension = 4^5 = 1024, too many
        # Use 3^5 = 243 grid points, plus some random
        grid_vals = [-4.0, 0.0, 4.0]
        grid_points = list(np.meshgrid(*([grid_vals]*dim)))
        grid_points = np.array(list(zip(*[g.flatten() for g in grid_points])))
        
        # Shuffle grid points
        rng.shuffle(grid_points)
        
        # Evaluate grid points
        for pt in grid_points:
            if eval_count >= budget or not time_ok():
                break
            counting_func(pt)
        
        # Add some random points for diversity
        n_random_initial = min(50, max(10, budget // 5))
        for _ in range(n_random_initial):
            if eval_count >= budget or not time_ok():
                break
            x_rand = rng.uniform(-5, 5, size=dim)
            counting_func(x_rand)

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
                best_x = np.zeros(dim)

        # Phase 2: Multi-start local optimization from top candidates
        # Sort by function value and pick top distinct regions
        if len(all_vals) > 0:
            sorted_idx = np.argsort(all_vals)
            top_candidates = []
            min_dist = 0.5  # Minimum distance between candidates
            
            for idx in sorted_idx:
                if len(top_candidates) >= 8:
                    break
                pt = all_points[idx]
                # Check distance to existing candidates
                is_distinct = True
                for tc in top_candidates:
                    if np.linalg.norm(pt - tc) < min_dist:
                        is_distinct = False
                        break
                if is_distinct:
                    top_candidates.append(pt)
            
            # Ensure best_x is in candidates
            if not any(np.allclose(tc, best_x) for tc in top_candidates):
                top_candidates.append(best_x.copy())
        else:
            top_candidates = [best_x.copy()]

        # Run Nelder-Mead from each candidate
        for x_start in top_candidates:
            if eval_count >= budget or not time_ok():
                break

            local_budget = budget - eval_count
            if local_budget < 15:
                break

            try:
                result = scipy.optimize.minimize(
                    counting_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(200, local_budget),
                        'maxfev': min(200, local_budget),
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                continue

        # Phase 3: CMA-ES for global refinement
        remaining = budget - eval_count
        if remaining > 30 and best_x is not None:
            try:
                popsize = min(15, max(8, remaining // 15))
                max_iter = max(3, remaining // popsize - 2)

                res = scipy.optimize.cma(
                    counting_func,
                    best_x,
                    sigma0=0.3,
                    options={
                        'maxiter': max_iter,
                        'popsize': popsize,
                        'tol': 1e-8,
                        'verbose': 0,
                        'bounds': [(-5.0, 5.0)] * dim
                    }
                )
            except Exception:
                pass

        # Phase 4: Final polish with tight Nelder-Mead
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(100, remaining),
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
