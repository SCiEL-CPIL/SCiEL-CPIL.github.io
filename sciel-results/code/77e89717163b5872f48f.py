"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s limit

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

        # Phase 1: Generate diverse starting points tailored for Rosenbrock
        # The Rosenbrock valley runs along y = x^2 - 1, so try points near this structure
        # but also random points since the function is rotated
        candidates = [np.zeros(dim)]
        
        # Try points along the diagonal (common for rotated Rosenbrock)
        for val in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
            candidates.append(np.full(dim, val))
        
        # Try points with alternating signs
        for val in [-2.0, 0.0, 2.0]:
            c = np.full(dim, val)
            c[::2] *= -1
            candidates.append(c)
        
        # Random points
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))
        
        # Try points near the known Rosenbrock minimum structure (all ones)
        candidates.append(np.ones(dim))
        candidates.append(np.ones(dim) * 0.5)
        candidates.append(np.ones(dim) * -1)

        # Evaluate candidates and track best
        candidate_results = []
        for c in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            f = safe_func(c)
            candidate_results.append((f, c.copy()))

        if not candidate_results:
            c = np.zeros(dim)
            f = safe_func(c)
            candidate_results.append((f, c.copy()))

        # Sort candidates by function value
        candidate_results.sort(key=lambda x: x[0])

        # Phase 2: Multi-start L-BFGS-B optimization
        # Allocate budget across starts: more budget for better starting points
        n_starts = min(len(candidate_results), 15)
        # Allocate maxfev proportional to remaining budget, weighted by candidate quality
        remaining_budget = budget - call_count
        
        # Use adaptive allocation: better starts get more budget
        # But ensure each start gets at least some budget
        min_fev = 50
        if n_starts * min_fev > remaining_budget:
            min_fev = max(10, remaining_budget // max(n_starts, 1))
        
        # Allocate remaining budget with weight based on candidate ranking
        weights = []
        for i in range(n_starts):
            # Earlier (better) candidates get more budget
            w = max(1, n_starts - i)
            weights.append(w)
        total_w = sum(weights)
        
        for i in range(n_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            
            f_init, x_init = candidate_results[i]
            
            # Allocate maxfev for this start
            maxfev = max(min_fev, int(remaining_budget * weights[i] / total_w))
            # Cap to not exceed remaining budget
            maxfev = min(maxfev, budget - call_count)
            
            if maxfev < 10:
                continue
            
            try:
                result = minimize(
                    scipy_wrapper,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': maxfev, 'maxfev': maxfev, 'gtol': 1e-8}
                )
                
                if result.x is not None and result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass

        # Phase 3: Final polish with Nelder-Mead on the best point if budget remains
        if call_count < budget - 20 and best_x is not None:
            if time.time() < deadline:
                remaining = budget - call_count
                try:
                    result = minimize(
                        scipy_wrapper,
                        best_x,
                        method='Nelder-Mead',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': remaining, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if result.x is not None and result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
