import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # 1. Initial Sampling
        n_init = 30
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return all_X

        F_init = np.array(all_F)
        f_min = np.min(F_init, axis=0)
        f_max = np.max(F_init, axis=0)
        
        # 2. Grid-based decomposition: partition weights into subproblems
        # Use more weights for better Pareto front coverage
        n_weights = 15
        weights = np.linspace(0.02, 0.98, n_weights)
        
        remaining_budget = budget - used
        evals_per_weight = max(5, int(remaining_budget / n_weights))
        
        def weighted_sum(x, w):
            if used >= budget or time.time() > deadline:
                return np.inf
            f = evaluate(x)
            if f is None:
                return np.inf
            return w[0] * f[0] + w[1] * f[1]

        # For each weight, find a good starting point and run local optimization
        for w_val in weights:
            if used >= budget or time.time() > deadline:
                break
            
            w_vec = np.array([w_val, 1.0 - w_val])
            
            # Find starting point: min weighted sum among current all_F
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                weighted_scores = F_arr[:, 0] * w_vec[0] + F_arr[:, 1] * w_vec[1]
                start_idx = np.argmin(weighted_scores)
                x_start = all_X[start_idx].copy()
            else:
                x_start = (lo + hi) / 2.0

            # Run Nelder-Mead with limited iterations
            max_iter = max(5, int(evals_per_weight / 2))
            
            stop_flag = [False]
            
            def callback(xk):
                nonlocal used
                if used >= budget - 2 or time.time() > deadline:
                    stop_flag[0] = True
                    return True
                return False

            try:
                res = minimize(
                    weighted_sum, 
                    x_start, 
                    args=(w_vec,), 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True},
                    callback=callback
                )
            except Exception:
                pass

        # 3. Final polish: refine Pareto-optimal points with tighter tolerances
        if used < budget and time.time() < deadline:
            # Find non-dominated points
            F_arr = np.array(all_F)
            n_points = len(F_arr)
            is_dominated = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                for j in range(n_points):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated[i] = True
                        break
            
            pareto_indices = np.where(~is_dominated)[0]
            
            # Polish top Pareto points with L-BFGS-B
            n_polish = min(5, len(pareto_indices))
            if n_polish > 0:
                # Sort by weighted sum with equal weights
                equal_w = np.array([0.5, 0.5])
                pareto_scores = F_arr[pareto_indices, 0] * equal_w[0] + F_arr[pareto_indices, 1] * equal_w[1]
                pareto_sorted = pareto_indices[np.argsort(pareto_scores)][:n_polish]
                
                for idx in pareto_sorted:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    x_start = all_X[idx].copy()
                    
                    def obj_func(x):
                        if used >= budget or time.time() > deadline:
                            return np.inf
                        f = evaluate(x)
                        if f is None:
                            return np.inf
                        return 0.5 * f[0] + 0.5 * f[1]
                    
                    try:
                        res = minimize(
                            obj_func,
                            x_start,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-8}
                        )
                    except Exception:
                        pass

        return all_X
