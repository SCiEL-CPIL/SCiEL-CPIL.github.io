"""Hybrid optimizer: coarse random search + L-BFGS-B local refinement."""
import numpy as np
import scipy.optimize
import time


class Optimizer:
    """Hybrid multi-start L-BFGS-B with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        d = self.dim
        rng = np.random.default_rng(42)

        # Track calls to enforce budget
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            calls_made += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Phase 1: Coarse random search to find diverse starting points
        n_random = 50
        candidates = []
        best_f = float('inf')
        best_x = None

        for i in range(n_random):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = safe_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if calls_made >= self.budget or time.time() > deadline:
            return best_x, best_f

        # Sort candidates by fitness and keep top ones
        candidates.sort(key=lambda t: t[0])
        
        # Decide how many local optimizations to run
        # We want to use remaining budget efficiently
        remaining_budget = self.budget - calls_made
        
        # Try to run local refinement on top candidates
        # L-BFGS-B typically uses ~2-5 function evaluations per iteration
        # Let's allocate budget among top candidates
        n_top = min(5, len(candidates))
        top_candidates = [c[1] for c in candidates[:n_top]]
        
        # Allocate remaining budget to local optimizations
        # Each L-BFGS-B run gets a portion of remaining budget
        budget_per_local = max(10, remaining_budget // n_top)
        
        # Phase 2: Local refinement using L-BFGS-B from top candidates
        for i, start_x in enumerate(top_candidates):
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Allocate slightly more budget to better starting points
            # But keep it simple: equal allocation for now
            maxfun = budget_per_local
            maxiter = max(5, maxfun // 2)  # Rough estimate
            
            try:
                res = scipy.optimize.minimize(
                    safe_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxfun': maxfun, 'maxiter': maxiter}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If L-BFGS-B fails, try a simpler local search
                pass

        # Phase 3: If we still have budget, do some additional random restarts
        # with short L-BFGS-B runs to explore more
        if calls_made < self.budget and time.time() < deadline:
            n_extra_restarts = min(10, (self.budget - calls_made) // 15)
            for _ in range(n_extra_restarts):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                
                # Random restart with small perturbation around best found
                if best_x is not None:
                    perturbation = 0.5 * rng.standard_normal(d)
                    x0 = np.clip(best_x + perturbation, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, d)
                
                maxfun = min(15, self.budget - calls_made)
                if maxfun < 5:
                    break
                    
                try:
                    res = scipy.optimize.minimize(
                        safe_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * d,
                        options={'maxfun': maxfun, 'maxiter': maxfun // 2}
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass

        # Ensure we have a valid result
        if best_x is None:
            # Should not happen if we made at least one call, but just in case
            x = rng.uniform(-5, 5, d)
            best_f = safe_func(x)
            best_x = x

        return best_x, best_f
