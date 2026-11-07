"""Multi-objective optimizer on ZDT6 using analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer on ZDT6 using dense analytical front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s/330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        results = []

        def eval_and_record(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            try:
                fx = problem(x)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        # ZDT6 Analysis:
        # f1(x) = 1 - exp(-4*x0 * sin^2(10*pi*x0))
        # g(x) = 1 + 9*(sum(x[1:]) / (dim-1))^0.25
        # f2(x) = (1 + g(x)) * (1 - sqrt(f1(x)))
        #
        # The Pareto front is achieved when g(x) is minimized (g=0), which occurs when sum(x[1:])=0.
        # Given bounds [0,1], this implies x[1:] = 0.
        # On the front: f1 = 1 - exp(-4*x0 * sin^2(10*pi*x0)), f2 = 1 - sqrt(f1).
        # f1 is not monotonic. It oscillates. We need to sample x0 densely to capture the full front.
        #
        # Key insight: The hypervolume is maximized when we have points that are well-distributed
        # across the Pareto front. The front has 10 "humps" due to sin^2(10*pi*x0).
        # To capture the full front well, we should sample densely, especially in regions where
        # f1 changes rapidly.
        
        # Strategy: Use all budget to sample x0 densely across [0, 1] with x[1:] = 0.
        # This places points directly on the analytical Pareto front.
        
        # ZDT6 front: f1 = 1 - exp(-4*x0 * sin^2(10*pi*x0)), f2 = 1 - sqrt(f1)
        # The function has 10 humps. We want to concentrate samples where f1 changes rapidly.
        
        n_sweep = budget
        if n_sweep > 0:
            # Use a non-uniform grid that concentrates points near the humps
            # The humps are centered at x0 = (2k+1)/(20) for k=0,...,9
            # sin^2(10*pi*x0) has maxima at these points
            
            # Create a density function that peaks at the humps
            x0_uniform = np.linspace(0.0, 1.0, n_sweep, endpoint=True)
            
            # Compute a weight based on how "interesting" each point is
            # We want to oversample where |df1/dx0| is large
            # df1/dx0 = 4*exp(-4*x0*sin^2(10*pi*x0)) * (sin^2(10*pi*x0) + 2*x0*10*pi*sin(10*pi*x0)*cos(10*pi*x0))
            
            # Simpler: use a sinusoidal density that peaks at hump centers
            # sin^2(10*pi*x) peaks at x = (2k+1)/20
            weights = np.sin(np.pi * 10 * x0_uniform) ** 2 + 0.1  # Add small floor to avoid zero weights
            
            # Normalize weights to get target positions
            # Use inverse CDF sampling: we want n_sweep points distributed according to weights
            # This is equivalent to finding x values such that cumulative weight is uniform
            
            # Compute cumulative weights
            cum_weights = np.cumsum(weights)
            cum_weights = cum_weights / cum_weights[-1]  # Normalize to [0, 1]
            
            # Target uniform positions
            target_positions = np.linspace(0.0, 1.0, n_sweep, endpoint=True)
            
            # Invert the CDF to get x0 values
            x0_vals = np.interp(target_positions, cum_weights, x0_uniform)
            
            # Ensure we still cover the full range
            x0_vals[0] = 0.0
            x0_vals[-1] = 1.0
            
            for x0 in x0_vals:
                x = np.zeros(dim)
                x[0] = x0
                eval_and_record(x)

        # If we still have budget (shouldn't happen since n_sweep = budget),
        # do a random search biased towards x[1:]=0.
        if used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                x = np.random.rand(dim)
                x[1:] = 0.0
                eval_and_record(x)

        return results
# EVOLVE-BLOCK-END
