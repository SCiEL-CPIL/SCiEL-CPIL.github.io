import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Sobol exploration + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return []
        
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_dominated[j] = True
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        # Phase 1: Sobol Exploration
        n_init = min(30, self.budget)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        X_init = lo + sobol.random(n_init) * range_
        
        for x in X_init:
            if time.time() > deadline - 5:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            eval_count += 1
            if eval_count >= self.budget:
                break

        if eval_count >= self.budget:
            return np.array(all_X)

        F_init = np.array(all_F)
        X_init_arr = np.array(all_X)
        
        # Find non-dominated points
        nd_indices = self._non_dominated_indices(F_init)
        if len(nd_indices) == 0:
            nd_indices = np.arange(len(F_init))
        
        X_nd = X_init_arr[nd_indices]
        F_nd = F_init[nd_indices]

        # Phase 2: Local Refinement
        n_local = 10
        remaining_budget = self.budget - eval_count
        if remaining_budget <= 0:
            return np.array(all_X)
            
        evals_per_local = remaining_budget // n_local
        if evals_per_local < 5:
            # Not enough budget for local search, just return what we have
            return np.array(all_X)

        # Generate weight vectors
        weights = []
        for w1 in np.linspace(0.1, 0.8, 4):
            for w2 in np.linspace(0.1, 0.8 - w1, 3):
                w3 = 1.0 - w1 - w2
                if w3 > 0.05:
                    weights.append(np.array([w1, w2, w3]))
        np.random.shuffle(weights)
        weights = weights[:n_local]
        
        if not weights:
            weights = [np.array([1/3, 1/3, 1/3])] * n_local

        for w in weights:
            if time.time() > deadline - 10:
                break
            if eval_count >= self.budget:
                break

            scores = F_nd @ w
            best_idx = np.argmin(scores)
            x0 = X_nd[best_idx]
            
            # Wrapper for scalar objective
            def scalar_obj(x, w=w):
                nonlocal eval_count
                if time.time() > deadline:
                    return 1e10
                if eval_count >= self.budget:
                    return 1e10
                
                fx = problem(x)
                all_X.append(x)
                all_F.append(fx)
                eval_count += 1
                return np.dot(fx, w)

            max_iter = max(1, evals_per_local // 5)
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass

        return np.array(all_X)
