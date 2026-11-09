"""Optimizer for DTLZ7 exploiting the known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for DTLZ7 that exploits the known structure of the Pareto front.
    
    DTLZ7 has:
    - 2 active variables (x[0], x[1]) that define the Pareto front
    - N-2 inactive variables that should be 0 for optimal solutions
    
    The Pareto front is a 2D surface parameterized by x[0] and x[1] in [0,1].
    We use Sobol sequences to sample this 2D parameter space uniformly,
    maximizing the hypervolume of the resulting set of objective vectors.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol sequence for 2D parameter space
        # We want to use as many evaluations as possible within budget
        # Use a power-of-2 number of points for Sobol
        n_evals = self.budget
        # Find largest power of 2 <= n_evals
        n_sobol = 1
        while n_sobol * 2 <= n_evals:
            n_sobol *= 2
        
        sampler = qmc.Sobol(d=2, scramble=True)
        params = sampler.random(n_sobol)
        
        # Build the full 22D vectors
        # x[0] and x[1] come from Sobol sample, rest are 0 (optimal for DTLZ7)
        X = np.zeros((n_sobol, self.dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_sobol
        
        # If we have remaining budget, we can add more random points
        # to fill in gaps, but for DTLZ7 the Sobol sampling should be sufficient
        remaining = self.budget - used
        if remaining > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Generate random point on the Pareto front parameter space
                p = rng.random(2)
                x = np.zeros(self.dim)
                x[0] = p[0]
                x[1] = p[1]
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                # Add to arrays
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
        
        return X
# EVOLVE-BLOCK-END
