# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using a deterministic space-filling lattice (Halton)
    for initial coverage, followed by targeted local refinement (Nelder-Mead) on
    extreme points (min f0, min f1, min f2) and the centroid (min sum) to maximize
    hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
        # First 'dim' primes
        primes = []
        candidate = 2
        while len(primes) < dim:
            is_prime = True
            for p in range(2, int(np.sqrt(candidate)) + 1):
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1

        points = np.zeros((n, dim))
        for d in range(dim):
            b = primes[d]
            f = 1.0 / b
            for i in range(1, n + 1):
                rad = i
                result = 0.0
                while rad > 0:
                    rad, rem = divmod(rad, b)
                    result += rem * f
                    f /= b
                points[i - 1, d] = result
        return points

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Step 1: Generate initial space-filling sample using Halton sequence
        # Use 60% of budget for initialization to ensure good coverage
        n_init = int(budget * 0.6)
        if n_init < 20:
            n_init = 20

        halton_points = self._halton(n_init, dim)
        X = lo + (hi - lo) * halton_points
        F = np.array([problem(x) for x in X])
        used = n_init

        # Step 2: Identify candidate points for refinement
        # We want to refine points that are best in specific objectives to capture the extremes
        # and the sum to capture the center.
        
        candidates = []
        
        # 1. Best point for each objective
        for j in range(n_obj):
            idx_min = np.argmin(F[:, j])
            candidates.append((idx_min, j)) # (index, objective_index_to_minimize)
        
        # 2. Best point for the sum of objectives (approximates center of Pareto front)
        sums = np.sum(F, axis=1)
        idx_sum = np.argmin(sums)
        candidates.append((idx_sum, -1)) # -1 indicates minimize sum

        # Step 3: Iteratively refine these candidates using Nelder-Mead
        # We have remaining budget. Let's distribute it among the candidates.
        remaining = budget - used
        if remaining > 0 and len(candidates) > 0:
            
            # Deduplicate candidates if any index appears multiple times
            unique_candidates = []
            seen = set()
            for idx, obj_idx in candidates:
                if idx not in seen:
                    seen.add(idx)
                    unique_candidates.append((idx, obj_idx))
            
            # If we have more candidates than we can afford to refine well, prioritize
            # Usually 3 objectives + 1 sum = 4 candidates.
            
            # Estimate cost per NM run. In 12D, NM is expensive.
            # Let's allocate remaining budget evenly.
            n_cands = len(unique_candidates)
            budget_per_run = max(10, remaining // n_cands)
            
            for idx, obj_idx in unique_candidates:
                if used >= budget:
                    break
                
                x0 = X[idx].copy()
                
                # Define scalar objective
                if obj_idx == -1:
                    def scalar_obj(x):
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        return np.sum(f)
                else:
                    # Capture obj_idx in closure
                    target_obj = obj_idx
                    def scalar_obj(x, _target=target_obj):
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        return f[_target]
                
                # Run Nelder-Mead
                try:
                    # maxiter is not exactly evals, but a good proxy.
                    # NM in dim D typically uses D+1 points.
                    # Let's limit maxiter to keep evals under budget_per_run
                    max_iter = max(5, budget_per_run // 2) 
                    
                    result = minimize(
                        scalar_obj,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    
                    x_new = np.clip(result.x, lo, hi)
                    f_new = problem(x_new)
                    used += result.nfev
                    
                    # Add the new point to the population
                    X = np.vstack([X, x_new])
                    F = np.vstack([F, f_new])
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
