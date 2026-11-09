"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with Sobol initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # --- Phase 1: Sobol-based global exploration ---
        # Generate Sobol points for structured coverage
        def sobol_points(n, dim, seed=42):
            """Generate n Sobol-like points in [0,1]^dim using a simple quasi-random sequence."""
            rng = np.random.default_rng(seed)
            # Use a Halton-like sequence for quasi-random coverage
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # use primes as bases: 2, 3, 5, 7, 11
                for i in range(n):
                    num = i + 1
                    result = 0.0
                    denom = 1.0
                    while num > 0:
                        num, rem = divmod(num, base)
                        denom *= base
                        result += rem / denom
                    points[i, d] = result
            return points

        # Generate 100 Sobol points for better global coverage
        n_global = 100
        sobol = sobol_points(n_global, dim)
        # Map to [-5, 5]^dim
        global_points = sobol * 10.0 - 5.0

        # Also include origin and a few random points
        extra_points = [np.zeros(dim)]
        rng = np.random.default_rng(123)
        for _ in range(10):
            extra_points.append(rng.uniform(-5, 5, dim))

        all_candidates = list(global_points) + extra_points

        best_x = None
        best_f = np.inf
        calls_used = 0
        eval_results = []

        for c in all_candidates:
            if time.time() > deadline:
                break
            fc = float(func(c))
            calls_used += 1
            eval_results.append((fc, c))
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        remaining = budget - calls_used

        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Sort candidates by their function values
        eval_results.sort(key=lambda x: x[0])
        
        # Select top K candidates for local refinement
        top_k = 10
        top_candidates = [c for f, c in eval_results[:top_k]]

        # --- Phase 2: Multi-start Nelder-Mead refinement ---
        # We will run Nelder-Mead from the top candidates.
        # Nelder-Mead is a derivative-free method suitable for black-box.
        
        # Determine max iterations per local search to stay within budget and time
        # Budget remaining is roughly 500 - 110 = 390.
        # If we do 10 restarts, we have ~39 evals each.
        # Nelder-Mead typically converges in < 50 evals in 5D if close.
        
        current_best_f = best_f
        current_best_x = best_x

        for x0 in top_candidates:
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Check if this start point is worse than current best by a lot, skip?
            # No, different basins might exist. But if it's significantly worse, 
            # Nelder-Mead might not climb out. 
            # However, f14 is unimodal-ish (convex-like structure with powers).
            
            try:
                # Use Nelder-Mead
                # maxiter: limit iterations. 
                # xatol/ftol: convergence criteria.
                result = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                # Check if we improved
                if result.fun < current_best_f:
                    current_best_f = float(result.fun)
                    current_best_x = np.clip(result.x, -5, 5)
                
                calls_used += result.nfev
                
            except Exception:
                # If an exception occurs, we might have used some calls.
                # It's hard to know exactly how many without tracking inside minimize.
                # To be safe, assume it used a few and break if we are close to limit.
                if calls_used > budget - 10:
                    break
                # Otherwise, continue? No, better to stop to avoid exceeding budget.
                break

        return current_best_x, current_best_f
# EVOLVE-BLOCK-END
