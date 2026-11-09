"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Deterministic Sobol Sequence Optimizer.
    Strategy:
      1. Pre-compute a large batch of Sobol sequence points (low-discrepancy sequence).
      2. These points provide maximal global coverage of the search space [-5, 5]^dim.
      3. Evaluate points sequentially. The deterministic nature ensures consistent
         progress and avoids the stochastic stagnation or local trapping risks 
         associated with evolutionary step-size adaptation (like (1+1)-ES).
      4. The early points in a Sobol sequence are well-distributed, which helps 
         with the AOCC metric that rewards early progress.
      5. Strict time guard ensures we never exceed the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0
        max_calls = self.budget

        # Generate Sobol sequence points
        # We generate more points than the budget to have a buffer, 
        # but we will only evaluate up to max_calls or until time runs out.
        # Sobol sequences are typically generated in powers of 2.
        # Let's generate enough points to cover the budget comfortably.
        # For dim=5, 512 points is a good power of 2.
        n_points = 1024 
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Sobol requires n to be a power of 2 for best properties, but works generally.
        # We'll generate 1024 points.
        points_unit = sampler.random(n_points)
        
        # Map from [0, 1]^d to [-5, 5]^d
        points = points_unit * 10.0 - 5.0

        # Iterate through the pre-computed points
        for i in range(len(points)):
            if calls >= max_calls:
                break
            if time.time() > deadline:
                break
                
            x = points[i]
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except Exception:
                # If an exception occurs, we should probably stop to be safe,
                # or just skip. The contract says "exception later keeps the calls made".
                # But to be safe and deterministic, if the function breaks, we stop.
                # However, the contract says "func(x) ... returns a float". 
                # Exceptions are unlikely if func is well-behaved. 
                # If it happens, we break to avoid infinite loops or repeated errors.
                break

        # If we haven't used all calls and time remains (unlikely with Sobol unless 
        # we ran out of points, which we won't with 1024 > 500), we could do a 
        # local polish, but the directive asks for a structural change to deterministic 
        # space-filling. Adding a stochastic polish might dilute the "pure" strategy 
        # test. However, a simple deterministic grid polish around the best point 
        # could be added if time permits. 
        # Given the "STAGNATION" directive, let's stick to the pure deterministic 
        # global search first to see if coverage is the key.
        
        # If best_x is still None (shouldn't happen), return a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
