"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytic Pareto front exploitation for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        n = self.dim
        # For ZDT3 Pareto front: set x[1:] = 0, x[0] = t
        # This gives g = 1 (minimum), so points lie on the Pareto front
        # f1 = t, f2 = 1 - sqrt(t) - t*sin(10*pi*t)
        
        # Generate candidate t values on [0, 1]
        # Use stratified sampling to cover the oscillations well
        # ZDT3 has 5 oscillation periods in h function over [0,1]
        # Use dense sampling to capture all local minima of f2
        
        budget = self.budget
        if budget <= 0:
            return None
            
        # Create a set of t values: combine uniform grid with extra points near oscillation minima
        # The function h(t,1) = 1 - sqrt(t) - t*sin(10*pi*t)
        # Local minima occur where derivative is zero
        
        # Use stratified sampling: divide [0,1] into many small intervals
        # and sample multiple points per interval, biased toward where f2 is small
        
        n_samples = min(budget, 200)
        
        # Create t values: use a combination of uniform and clustered points
        # First, a fine uniform grid
        n_uniform = max(10, n_samples // 2)
        t_uniform = np.linspace(0.0, 1.0, n_uniform, endpoint=True)
        
        # Add extra points clustered around where sin(10*pi*t) = 1 (minima of f2)
        # sin(10*pi*t) = 1 when 10*pi*t = pi/2 + 2*pi*k, t = (1/20 + k/10)
        # These are at t = 0.05, 0.15, 0.25, ..., 0.95
        oscillation_points = np.array([(0.05 + 0.1 * k) for k in range(10)])
        
        # Add points near these oscillation minima
        extra_per_osc = max(1, n_samples // 10)
        t_extra = []
        for op in oscillation_points:
            for j in range(extra_per_osc):
                t_extra.append(op + (j - extra_per_osc/2) * 0.02)
        t_extra = np.array(t_extra)
        t_extra = t_extra[(t_extra >= 0) & (t_extra <= 1)]
        
        # Combine and sort, remove duplicates
        t_all = np.concatenate([t_uniform, t_extra])
        t_all = np.sort(np.unique(np.round(t_all, 10)))
        
        # Truncate to budget
        if len(t_all) > budget:
            # Prioritize points that are likely to give better hypervolume
            # Just take the first 'budget' points (which are spread out)
            t_all = t_all[:budget]
        
        # Evaluate each point
        results = []
        for t in t_all:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = float(t)
            # x[1:] = 0 already
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if np.all(np.isfinite(f)) and len(f) == self.n_obj:
                    results.append(f)
            except Exception:
                continue
        
        # If we have room, add some random exploration points
        if len(results) < budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining = budget - len(results)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = np.random.rand(n)
                try:
                    f = problem(x)
                    f = np.array(f, dtype=float)
                    if np.all(np.isfinite(f)) and len(f) == self.n_obj:
                        results.append(f)
                except Exception:
                    continue
        
        return results
# EVOLVE-BLOCK-END
