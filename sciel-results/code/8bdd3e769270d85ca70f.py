"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Multi-start L-BFGS-B optimizer for BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 330s limit

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def scipy_wrapper(x):
            nonlocal call_count, deadline
            if call_count >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            return safe_func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Sparse Initial Sampling
        # Reduce initial evaluations to conserve budget for refinement
        # Focus on structurally relevant points for Rosenbrock
        candidates = []
        candidates.append(np.ones(dim))       # Unrotated minimum
        candidates.append(np.zeros(dim))       # Center
        # Add a few points along the unrotated valley structure
        for x0 in [-2.0, 0.0, 2.0]:
            c = np.zeros(dim)
            c[0] = x0
            for i in range(1, dim):
                c[i] = c[i-1]**2 - 1
            c = np.clip(c, -5, 5)
            candidates.append(c)
        # Add random points
        for _ in range(5):
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        # Evaluate initial candidates
        candidate_results = []
        for c in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            c_clipped = np.clip(c, -5.0, 5.0)
            f = safe_func(c_clipped)
            candidate_results.append((f, c_clipped.copy()))

        if not candidate_results:
            c = np.zeros(dim)
            f = safe_func(c)
            candidate_results.append((f, c.copy()))

        # Sort by function value
        candidate_results.sort(key=lambda x: x[0])
        
        # Phase 2: Adaptive Refinement Loop
        # Instead of running all starts, we refine the best one, and if it fails to improve significantly,
        # we try the next best or generate new randoms.
        
        # We will iterate through the initial candidates, but with a "greedy" approach:
        # If the current best improves significantly, we stick with it and polish.
        # If not, we try the next candidate.
        
        # To maximize AOCC, we want to find a good point quickly.
        # Let's try the top 3-4 candidates with substantial budget each, but stop early if we find a very good one.
        
        max_starts_to_try = min(4, len(candidate_results))
        
        for i in range(max_starts_to_try):
            if call_count >= budget or time.time() > deadline:
                break
            
            f_init, x_init = candidate_results[i]
            
            # If this start is much worse than the current best, skip it?
            # Actually, candidate_results is sorted, so x_init is the i-th best initial point.
            # If best_f is already very low, and f_init is high, the L-BFGS might still find the minimum
            # if it's in the basin of attraction. But for Rosenbrock, basins are tricky.
            
            # Allocate a decent chunk of remaining budget to this start
            # We want to allow it to converge.
            remaining = budget - call_count
            # Cap the budget per start to ensure we don't waste everything on one bad start
            # but also don't starve a good one.
            maxfev = min(remaining, max(50, int(remaining * 0.5)))
            
            if maxfev < 10:
                continue
            
            try:
                result = minimize(
                    scipy_wrapper,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': maxfev,
                        'maxfev': maxfev,
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
                
                if result.x is not None and result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
                    
            except Exception:
                pass

        # Phase 3: Intensive Polish on the Best Found
        # Now that we have a good best_x, use remaining budget to polish it.
        if call_count < budget - 20 and best_x is not None:
            if time.time() < deadline:
                remaining = budget - call_count
                try:
                    # High precision L-BFGS-B
                    result = minimize(
                        scipy_wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'gtol': 1e-12,
                            'ftol': 1e-15
                        }
                    )
                    if result.x is not None and result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 4: Final Nelder-Mead Polish
        # NM can sometimes escape local minima or refine in a different way
        if call_count < budget - 10 and best_x is not None:
            if time.time() < deadline:
                remaining = budget - call_count
                try:
                    result = minimize(
                        scipy_wrapper,
                        best_x,
                        method='Nelder-Mead',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.x is not None and result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
