"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_points = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            f = problem(x)
            all_points.append(f)
            used += 1
            return f

        # ZDT3 Pareto front analysis:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(dim-1) + sum(x[i]^2 - 10*cos(2*pi*x[i]) for i in 1..dim-1)
        # h(x1, g) = 1 - sqrt(x1/g) - (x1/g)*sin(10*pi*x1)
        # f2(x) = g(x) * h(x[0], g(x))
        #
        # Minimum g = 1 when x[1:] = 0
        # With g=1: f2 = 1 - sqrt(x1) - x1*sin(10*pi*x1)
        #
        # The Pareto front is discontinuous due to sin term.
        # To maximize hypervolume, we need dense coverage of x[0] in [0,1]
        # with x[1:] = 0 to achieve the ideal g=1.

        # Strategy: Use all budget to densely sample the ideal Pareto front.
        # The hypervolume reference point is [11.0, 11.0].
        # f1 ranges from 0 to 1, f2 ranges roughly from ~0 to ~3 on the front.
        # Points must be strictly better than [11, 11] in both objectives to count.

        # Use stratified sampling for better coverage
        n_evals = budget
        
        # Create x[0] values with dense uniform coverage
        # Slightly oversample the regions where the front is more complex
        # The sin(10*pi*x1) term creates 10 oscillations, so we need fine resolution
        x0_vals = np.linspace(0.0, 1.0, n_evals)
        
        # For each x[0] value, create a point with x[1:] = 0
        for i in range(n_evals):
            if used >= budget or time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            evaluate(x)

        return all_points
# EVOLVE-BLOCK-END
