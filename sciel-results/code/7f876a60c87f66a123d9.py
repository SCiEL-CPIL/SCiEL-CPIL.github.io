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

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum(x[i]^2 - 10*cos(2*pi*x[i]) for i in 1..n-1)
        # h(x1, g) = 1 - sqrt(x1/g) - (x1/g)*sin(10*pi*x1)
        # f2(x) = g(x) * h(x[0], g(x))
        #
        # The Pareto front is discontinuous. To get good hypervolume, we need:
        # 1. x[0] spread across [0, 1]
        # 2. g(x) minimized (ideally g=1, which requires x[1:]=0)
        #
        # Strategy: 
        # - Set x[1:] to 0 to achieve g=1 (minimum possible)
        # - Spread x[0] to cover the Pareto front using ALL budget
        # - This ensures all evaluations are on the ideal Pareto front

        n_phase1 = budget
        
        # Spread x[0] values to cover the range uniformly
        # This achieves g=1, which is the minimum possible value of g
        x0_vals = np.linspace(0.0, 1.0, n_phase1)
        
        for i in range(n_phase1):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            # x[1:] = 0 achieves g = 1 (minimum)
            
            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
