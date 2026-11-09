import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.low = -5.0
        self.high = 5.0

    def _halton(self, n, dim):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = primes[d]
            f = 1.0 / b
            i_val = np.arange(1, n + 1)
            frac = np.zeros(n)
            while np.any(i_val > 0):
                frac += f * (i_val % b)
                i_val //= b
                f /= b
            points[:, d] = frac
        return points

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        low, high = self.low, self.high
        calls_made = 0
        
        def count_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Phase 1: Coarse Search
        n_coarse = 60
        halton_coarse = self._halton(n_coarse, dim)
        x_coarse = low + (high - low) * halton_coarse
        
        best_x = None
        best_f = np.inf
        coarse_results = []
        
        for i in range(n_coarse):
            if calls_made >= budget: break
            x = x_coarse[i]
            fx = count_call(x)
            coarse_results.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort coarse results to get top candidates
        coarse_results.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement on Top Candidates
        # We have budget - n_coarse left.
        # Let's pick top 3 candidates.
        n_candidates = min(3, len(coarse_results))
        remaining_budget = budget - calls_made
        
        # Budget per candidate
        budget_per_candidate = remaining_budget // n_candidates
        
        for i in range(n_candidates):
            if calls_made >= budget: break
            start_x = coarse_results[i][1].copy()
            start_f = coarse_results[i][0]
            
            # Successive Halving Search
            center = start_x.copy()
            radius = 2.0
            local_best_f = start_f
            local_best_x = start_x.copy()
            
            # We will do a few iterations of shrinking
            # Each iteration samples ~10-20 points
            # Total budget for this candidate is budget_per_candidate
            # Let's allocate 15 points per iteration
            points_per_iter = 15
            
            while calls_made + points_per_iter <= budget and radius > 1e-3:
                halton_local = self._halton(points_per_iter, dim)
                # Map to [center - radius, center + radius]
                x_local = center + radius * (2 * halton_local - 1)
                x_local = np.clip(x_local, low, high)
                
                iter_best_f = local_best_f
                iter_best_x = local_best_x
                
                for j in range(points_per_iter):
                    if calls_made >= budget: break
                    x = x_local[j]
                    fx = count_call(x)
                    if fx < iter_best_f:
                        iter_best_f = fx
                        iter_best_x = x.copy()
                
                # Update local best
                if iter_best_f < local_best_f:
                    local_best_f = iter_best_f
                    local_best_x = iter_best_x.copy()
                    center = local_best_x.copy()
                    radius *= 0.5
                else:
                    # No improvement, shrink radius anyway to converge or break?
                    # If no improvement, shrinking might just stay in same basin.
                    # But we must respect "refining step size by half each time a new global minimum is found".
                    # If no new min found, we might stop refining this candidate to save budget for others?
                    # Or just stop.
                    break
                
                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

        return best_x, best_f
