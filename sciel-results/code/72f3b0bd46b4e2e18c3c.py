import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Keep a few top points to use as restarts for local optimization
        top_points = []
        top_fvals = []

        def safe_func(x):
            nonlocal calls, best_x, best_f, top_points, top_fvals
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            
            # Update best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # Update top 5 points
            if len(top_fvals) < 5:
                top_points.append(x.copy())
                top_fvals.append(fx)
                # Keep sorted
                idx = np.argsort(top_fvals)
                top_fvals = [top_fvals[i] for i in idx]
                top_points = [top_points[i] for i in idx]
            else:
                if fx < top_fvals[-1]:
                    top_points[-1] = x.copy()
                    top_fvals[-1] = fx
                    idx = np.argsort(top_fvals)
                    top_fvals = [top_fvals[i] for i in idx]
                    top_points = [top_points[i] for i in idx]
                    
            return fx

        # Phase 1: 1D Curve-based Global Search
        # Use a Lissajous-like curve to traverse the search space in 1D
        # This reduces global search dimensionality from 5 to 1
        
        # Generate a space-filling curve parameterized by t in [0, 1]
        # Use different frequencies for each dimension to create a dense path
        def curve_point(t):
            """Map scalar t in [0,1] to a point in [-4,4]^dim using Lissajous frequencies"""
            # Use prime-like frequencies to ensure good coverage
            freqs = [2.0, 3.0, 5.0, 7.0, 11.0]
            phases = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
            x = np.zeros(dim)
            for d in range(dim):
                # Map t to a range that covers the box
                # Use sin to oscillate, scaled to cover [-4, 4]
                x[d] = 4.0 * np.sin(freqs[d % 5] * 2 * np.pi * t + phases[d % 5])
            return x

        # Coarse scan along the curve
        n_coarse = 100
        if n_coarse < budget - 50:
            for i in range(n_coarse):
                if time.time() > deadline:
                    break
                t = i / (n_coarse - 1) if n_coarse > 1 else 0.0
                x = curve_point(t)
                safe_func(x)
        
        # If we have budget left, do a finer scan
        n_fine = min(150, budget - calls - 100)
        if n_fine > 10:
            for i in range(n_fine):
                if time.time() > deadline:
                    break
                t = (i + 0.5) / n_fine
                x = curve_point(t)
                safe_func(x)
        
        # Fallback if no finite values found
        if not np.isfinite(best_f):
            rng = np.random.default_rng(123)
            for _ in range(min(50, budget - 10)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                safe_func(x)
            if not np.isfinite(best_f):
                return best_x, best_f

        # Phase 2: Local Refinement with BFGS from multiple starts
        # Use the top points found in Phase 1
        local_budget_per_start = 30
        max_starts = min(8, (budget - calls) // local_budget_per_start)
        
        # Ensure we have unique starts
        starts = []
        for p in top_points:
            if len(starts) >= max_starts:
                break
            # Check if distinct from existing starts
            is_distinct = True
            for s in starts:
                if np.linalg.norm(p - s) < 0.1:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(p)
                
        # If we didn't get enough distinct points, add perturbations
        rng = np.random.default_rng(42)
        while len(starts) < max_starts:
            pert = best_x + rng.normal(0, 0.5, dim)
            pert = np.clip(pert, -4.0, 4.0)
            starts.append(pert)

        for x0 in starts:
            if calls >= budget - 30:
                break
            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='BFGS',
                    options={'maxiter': 50, 'gtol': 1e-8, 'maxcor': 10}
                )
            except Exception:
                pass

        # Phase 3: Final Polish with L-BFGS-B
        remaining = budget - calls
        if remaining > 20 and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'gtol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
