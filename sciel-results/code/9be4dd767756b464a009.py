import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

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
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # 1. Initial Sampling - use structured sampling for better coverage
        n_init = 60
        # Use Sobol sequence for quasi-random uniform distribution
        sampler = qmc.Sobol(d=dim, scramble=True, seed=0)
        points = sampler.random(n_init)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * points[i]
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return all_X

        # 2. Decomposition via Weighted Sum
        # Use Nelder-Mead which is more robust for small budgets and low dimensions
        # without finite difference overhead
        n_weights = 20
        weights = np.linspace(0.02, 0.98, n_weights)
        
        # Remaining budget for local optimization
        remaining_budget = budget - used
        # Reserve budget for polish
        polish_budget = 20
        available_for_weights = remaining_budget - polish_budget
        if available_for_weights < 0:
            available_for_weights = 0
        evals_per_weight = max(10, available_for_weights // n_weights)

        def weighted_sum(x, w):
            if used >= budget or time.time() > deadline:
                return np.inf
            f = evaluate(x)
            return w[0] * f[0] + w[1] * f[1]

        # For each weight, find a good starting point and run L-BFGS-B
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

            # Run Nelder-Mead with tight budget control
            stop_flag = [False]
            
            def callback(xk):
                nonlocal used
                if used >= budget - polish_budget or time.time() > deadline:
                    stop_flag[0] = True
                    return True
                return False

            try:
                res = minimize(
                    weighted_sum, 
                    x_start, 
                    args=(w_vec,), 
                    method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6},
                    callback=callback
                )
            except Exception:
                pass

        # 3. Final polish: refine top Pareto points with tighter tolerances
        if used < budget - 5 and len(all_F) > 5:
            F_arr = np.array(all_F)
            # Find non-dominated points
            pareto_mask = np.ones(len(all_F), dtype=bool)
            for i in range(len(all_F)):
                if not pareto_mask[i]:
                    continue
                for j in range(len(all_F)):
                    if i == j or not pareto_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        pareto_mask[i] = False
                        break
            
            pareto_indices = np.where(pareto_mask)[0]
            # Sort by first objective and take top 5
            if len(pareto_indices) > 5:
                pareto_indices = pareto_indices[np.argsort(F_arr[pareto_indices, 0])][:5]
            elif len(pareto_indices) > 0:
                pareto_indices = pareto_indices[:5]
            
            for idx in pareto_indices:
                if used >= budget - 2 or time.time() > deadline:
                    break
                
                x_start = all_X[idx].copy()
                
                # Polish using Nelder-Mead with tighter tolerance
                def polish_objective(x):
                    if used >= budget or time.time() > deadline:
                        return np.inf
                    f = evaluate(x)
                    # Use a small epsilon to avoid exact ties
                    return f[0] + 1e-6 * f[1]
                
                stop_flag2 = [False]
                def callback2(xk):
                    nonlocal used
                    if used >= budget - 1 or time.time() > deadline:
                        stop_flag2[0] = True
                        return True
                    return False
                
                try:
                    minimize(
                        polish_objective, 
                        x_start, 
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8},
                        callback=callback2
                    )
                except Exception:
                    pass

        return all_X
