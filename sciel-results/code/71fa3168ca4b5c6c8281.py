"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid L-BFGS-B + (1+1)-ES optimizer for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        dim = self.dim
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Generate diverse starting points using a coarse random search
        # The step ellipsoidal function has plateaus, so we need to find
        # points close to the minimum first, then exploit with L-BFGS-B
        
        # Try origin first
        origin = np.zeros(dim)
        f_origin = _eval(origin)
        if f_origin is not None:
            pass  # best_x, best_f already updated

        # Generate diverse starting points in [-4, 4]^5 (minimizer is in this range)
        n_starts = 20
        start_points = [origin.copy()]
        
        for i in range(n_starts - 1):
            # Mix of uniform random and structured points
            if i < 5:
                # Structured: random sign patterns with varying magnitudes
                signs = rng.choice([-1, 1], size=dim)
                magnitudes = rng.uniform(0.5, 4.0, size=dim)
                x = signs * magnitudes
            else:
                # Uniform random in [-4, 4]
                x = rng.uniform(-4, 4, size=dim)
            x = np.clip(x, -5, 5)
            start_points.append(x)

        # Evaluate all starting points
        for x in start_points[1:]:
            _eval(x)

        # Sort starting points by fitness and keep top ones
        if best_x is not None:
            start_points_sorted = sorted(start_points, key=lambda x: _get_cached_f(x) if False else best_f)
        
        # Phase 2: Run L-BFGS-B from the best starting points
        # Use the top few starting points for L-BFGS-B
        # We'll run L-BFGS-B from multiple points, stopping if time runs out
        
        # Keep track of all evaluated points to avoid re-evaluation
        evaluated = {}
        
        def _eval_cached(x, tol=1e-10):
            nonlocal calls, best_x, best_f
            # Check if we've already evaluated this point
            key = tuple(np.round(x, 8))
            if key in evaluated:
                return evaluated[key]
            
            if calls >= self.budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            evaluated[key] = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Re-evaluate starting points with caching
        for x in start_points:
            key = tuple(np.round(x, 8))
            if key not in evaluated:
                _eval_cached(x)

        # Sort by actual function values
        start_points_with_f = []
        for x in start_points:
            key = tuple(np.round(x, 8))
            if key in evaluated:
                start_points_with_f.append((x, evaluated[key]))
        
        start_points_with_f.sort(key=lambda t: t[1])
        
        # Run L-BFGS-B from top 5 starting points
        n_lbfgs = 5
        for idx in range(min(n_lbfgs, len(start_points_with_f))):
            if calls >= self.budget or time.time() > deadline - 30:
                break
                
            x0 = start_points_with_f[idx][0].copy()
            f0 = start_points_with_f[idx][1]
            
            def lbfgs_func(x):
                nonlocal calls, best_f, best_x
                if calls >= self.budget or time.time() > deadline:
                    return best_f
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            try:
                res = minimize(lbfgs_func, x0, method='L-BFGS-B',
                              bounds=[(-5, 5)] * dim,
                              options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Use remaining budget for (1+1)-ES refinement around best point
        if calls < self.budget and time.time() < deadline - 10:
            sigma_es = 0.5
            x_es = best_x.copy() if best_x is not None else np.zeros(dim)
            f_es = best_f
            
            # Adaptive step size
            no_improve = 0
            max_no_improve = 15
            
            while calls < self.budget and time.time() < deadline:
                if no_improve > max_no_improve:
                    break
                
                # Generate candidate
                z = rng.standard_normal(dim)
                x_cand = x_es + sigma_es * z
                x_cand = np.clip(x_cand, -5, 5)
                
                fx = _eval_cached(x_cand)
                if fx is None:
                    break
                
                if fx < f_es:
                    x_es = x_cand.copy()
                    f_es = fx
                    no_improve = 0
                    sigma_es *= 1.1  # increase step size on success
                else:
                    no_improve += 1
                    sigma_es *= 0.9  # decrease step size on failure
                
                sigma_es = np.clip(sigma_es, 1e-6, 5.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
