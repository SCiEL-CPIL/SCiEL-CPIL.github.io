"""Direct approach for BBOB f22 (Gallagher 21 peaks) optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Adaptive multi-start optimizer for BBOB f22 using CMA-ES-like strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(7)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Broad exploration with stratified sampling
        # Use Latin Hypercube for good coverage of the 5D domain
        n_lhs = min(100, budget)
        
        # Simple stratified sampling: divide each dimension into bins
        # For 5D with ~100 points, use ~3 bins per dimension (3^5=243, too many)
        # Use 2 bins per dimension: 2^5=32 strata, ~3 points per stratum
        n_strata = 32
        points_per_stratum = max(1, n_lhs // n_strata)
        
        initial_points = []
        # Generate stratified points
        for s in range(n_strata):
            # Decode stratum index to bin indices
            bins = []
            temp = s
            for d in range(dim):
                bins.append(temp % 2)
                temp //= 2
            bins = np.array(bins)
            
            for _ in range(points_per_stratum):
                # Uniform random within the stratum
                pt = np.zeros(dim)
                for d in range(dim):
                    if bins[d] == 0:
                        pt[d] = rng.uniform(-5.0, 0.0)
                    else:
                        pt[d] = rng.uniform(0.0, 5.0)
                initial_points.append(pt)
        
        # Add center and some structured points
        initial_points.append(np.zeros(dim))
        for d in range(dim):
            for v in [-3.0, -1.5, 1.5, 3.0]:
                pt = np.zeros(dim)
                pt[d] = v
                initial_points.append(pt)
        
        # Evaluate all initial points
        for x in initial_points:
            if not time_ok():
                break
            eval_and_track(x)
        
        if not time_ok():
            return best_x, best_f

        # Phase 2: Local refinement using Nelder-Mead from best point
        # and several other promising points
        # We need to identify promising points, but we didn't track them individually
        # So we'll restart from best_x and a few random nearby points
        
        # First, do a local refinement from the best point
        if time_ok():
            try:
                res = minimize(
                    eval_and_track,
                    best_x.copy(),
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass
        
        if not time_ok():
            return best_x, best_f

        # Phase 3: Multi-start local search
        # Generate several starting points around the best found region
        # and do local optimization from each
        
        n_starts = 8
        sigma = 1.5  # Initial spread
        
        for i in range(n_starts):
            if not time_ok():
                break
            
            # Generate a starting point: mix of best_x + noise and random
            if i % 2 == 0:
                # Local perturbation of best
                x0 = best_x + rng.normal(0, sigma, size=dim)
            else:
                # Random point in a box around best
                x0 = best_x + rng.uniform(-sigma, sigma, size=dim)
            
            x0 = np.clip(x0, -5.0, 5.0)
            
            try:
                res = minimize(
                    eval_and_track,
                    x0,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass
        
        if not time_ok():
            return best_x, best_f

        # Phase 4: Adaptive restart with shrinking sigma
        # If we still have budget, do more local searches with smaller sigma
        sigma = 0.5
        n_more_starts = 6
        for i in range(n_more_starts):
            if not time_ok():
                break
            
            x0 = best_x + rng.normal(0, sigma, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)
            
            try:
                res = minimize(
                    eval_and_track,
                    x0,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
            except Exception:
                pass
        
        # Phase 5: Final polish with L-BFGS-B if budget remains
        if time_ok() and calls < budget - 10:
            try:
                res = minimize(
                    eval_and_track,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 30}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
