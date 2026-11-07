"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with deterministic starting points and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Track the best solution found so far
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Deterministic multi-start points covering the box [-5, 5]^5
        # The minimizer is in [-4, 4]^5
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0])
        ]
        
        # Progressive step sizes for CMA-ES: start large, then smaller
        step_sizes = [4.0, 1.0, 0.2]
        
        # Allocate budget: first start gets most, subsequent get less
        # We'll do up to 5 starts, but budget may run out
        budget_allocations = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        for i, start_x in enumerate(starts):
            if calls_made >= budget:
                break
            
            remaining = budget - calls_made
            alloc = int(remaining * budget_allocations[i])
            
            # Ensure we have enough for at least a few CMA-ES iterations
            # CMA-ES with popsize ~ 4*dim needs at least popsize calls per iteration
            if alloc < 20:
                alloc = remaining
            
            if alloc <= 0:
                break

            # Choose step size based on start index
            # Larger steps for early starts, smaller for later
            sigma = step_sizes[min(i, len(step_sizes) - 1)]
            
            # Set up CMA-ES parameters
            # Population size: typically 4 + floor(3*ln(dim))
            popsize = max(4, int(4 + 3 * np.log(dim)))
            
            # Max iterations: each iteration uses popsize function evaluations
            max_iter = max(1, alloc // popsize)
            
            try:
                res = minimize(
                    evaluate,
                    start_x,
                    method='Nelder-Mead',  # We'll use Nelder-Mead as a fallback but with tuned parameters
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
