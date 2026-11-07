import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        # Track all evaluated points and their objectives
        X_all = []
        F_all = []
        eval_count = 0

        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            try:
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                f = np.array(f, dtype=float)
                if np.all(np.isfinite(f)):
                    X_all.append(x_clipped)
                    F_all.append(f)
                    eval_count += 1
                    return f
            except Exception:
                pass
            return None

        # 1. Initial random sampling
        n_init = 30
        for _ in range(n_init):
            if time.time() > deadline or eval_count >= self.budget:
                break
            x = lo + np.random.rand(self.dim) * span
            safe_eval(x)

        if eval_count >= self.budget or time.time() > deadline:
            return np.array(X_all) if X_all else np.zeros((1, self.dim))

        # 2. Differential Evolution for each objective separately
        # This helps find good points along each objective's Pareto front
        # We use small populations to stay within budget
        for obj_idx in range(self.n_obj):
            if time.time() > deadline or eval_count >= self.budget:
                break

            def single_obj_func(x):
                f = safe_eval(x)
                if f is None:
                    return 1e10
                return f[obj_idx]

            # Run DE with small population
            try:
                res = differential_evolution(
                    single_obj_func,
                    bounds=list(zip(lo, hi)),
                    maxiter=50,
                    popsize=10,
                    tol=0.01,
                    seed=42 + obj_idx,
                    polish=False
                )
            except Exception:
                pass

        if eval_count >= self.budget or time.time() > deadline:
            return np.array(X_all) if X_all else np.zeros((1, self.dim))

        # 3. Local refinement using Nelder-Mead on a weighted sum
        # Start from the best points found so far
        if F_all:
            F_arr = np.array(F_all)
            X_arr = np.array(X_all)

            # Normalize objectives to [0,1] for weighting
            f_min = F_arr.min(axis=0)
            f_max = F_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0

            # Pick a few starting points for local search
            # Use non-dominated points if possible, else random
            # Simple non-dominated check
            nd_indices = []
            for i in range(len(F_arr)):
                dominated = False
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)

            if nd_indices:
                start_indices = nd_indices[:10]  # Take up to 10 non-dominated points
            else:
                start_indices = np.random.choice(len(F_arr), size=min(10, len(F_arr)), replace=False)

            for idx in start_indices:
                if time.time() > deadline or eval_count >= self.budget:
                    break

                x0 = X_arr[idx]

                # Weighted sum objective
                # Try a few different weight vectors
                weights = [
                    np.array([1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.5, 0.5]),
                    np.array([0.8, 0.2]),
                    np.array([0.2, 0.8])
                ]

                for w in weights:
                    if time.time() > deadline or eval_count >= self.budget:
                        break

                    def weighted_func(x):
                        f = safe_eval(x)
                        if f is None:
                            return 1e10
                        f_norm = (f - f_min) / f_range
                        return np.dot(w, f_norm)

                    try:
                        res = minimize(
                            weighted_func,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
                        )
                    except Exception:
                        pass

        # Return the last point or a dummy array (return value is ignored)
        if X_all:
            return np.array(X_all[-1])
        else:
            return np.zeros(self.dim)
