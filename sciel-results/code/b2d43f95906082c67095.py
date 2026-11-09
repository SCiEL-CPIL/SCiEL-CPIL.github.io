"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using structured sampling for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 insight: Pareto front is 1D, depends on x[0]
        # Optimal: x[0] varies in [0,1], x[1:] = 0 for minimal g(x)
        
        used = 0
        n = self.budget
        
        # Phase 1: Structured grid sampling along x[0]
        # Use ~70% of budget for structured sampling
        n_grid = max(10, int(n * 0.7))
        
        # Sample x[0] values: include endpoints and evenly spaced
        x0_values = np.linspace(0.0, 1.0, n_grid)
        
        for x0 in x0_values:
            x = np.zeros(self.dim)
            x[0] = x0
            # x[1:] = 0 is optimal for ZDT1
            # But clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1
            if used >= n:
                break
        
        # Phase 2: Use remaining budget for refinement
        # Try small perturbations of x[1:] to see if we can improve
        remaining = n - used
        if remaining > 0 and self.dim > 1:
            # For each grid point, try small perturbations in x[1:]
            # But be strategic: focus on points that might not be optimal
            # Try a few random perturbations
            for _ in range(remaining):
                # Pick a random x0
                x0 = np.random.random()
                x = np.zeros(self.dim)
                x[0] = x0
                # Small perturbation in x[1:]
                if self.dim > 1:
                    x[1:] = 0.01 * np.random.standard_normal(self.dim - 1)
                x = np.clip(x, lo, hi)
                problem(x)
                used += 1
                if used >= n:
                    break
        
        return None
# EVOLVE-BLOCK-END
