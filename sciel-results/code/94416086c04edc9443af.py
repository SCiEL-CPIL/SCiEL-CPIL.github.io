"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol sequence based multi-objective search for uniform coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol sequence points
        # Sobol requires power of 2 for sample size for efficiency/validity in some implementations, 
        # but scipy handles arbitrary N by generating next_power_of_2 and truncating.
        # We want exactly self.budget points.
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate enough points, then take the first self.budget
        # Note: Sobol sequence is deterministic but scrambled for randomness
        points_unit = sampler.random(n=self.budget)
        
        # Map to bounds
        X = lo + points_unit * (hi - lo)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Time guard: if we exceeded time, we still have results, but we should stop generating more if we were in a loop.
        # Here we generate all at once. If problem evaluation is slow, this might timeout.
        # WFG9 is fast, so 200 evaluations should be well under 300s.
        
        return X
# EVOLVE-BLOCK-END
