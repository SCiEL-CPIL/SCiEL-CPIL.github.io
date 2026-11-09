"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Hybrid optimizer: multi-restart L-BFGS-B + differential evolution for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def make_eval(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if x.shape != (dim,):
                x = x.reshape(-1)[:dim]
            fx = float(func(x))
            calls_made += 1
            return fx

        # Phase 1: Differential Evolution for global exploration
        # Use a subset of budget for DE
        de_budget = min(200, budget // 2)
        if calls_made < budget and time.time() < deadline:
            try:
                bounds = [(-5.0, 5.0)] * dim
                # Use a wrapper that respects our call budget
                def de_func(x):
                    return make_eval(x)

                # Run DE with limited maxiter and popsize
                # popsize * dim evaluations per generation
                popsize = 10
                maxiter = max(1, de_budget // (popsize * dim))
                
                # Initial population from random points
                x0 = rng.uniform(-5, 5, size=(popsize, dim))
                
                # Use differential_evolution with our custom function
                # maxfees controls total function evaluations
                result = differential_evolution(
                    de_func, bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-6,
                    seed=int(rng.integers(0, 2**31)),
                    maxfees=de_budget - calls_made,
                    polish=False  # We'll do our own polish
                )
                
                if result.x is not None and result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 2: Multi-restart L-BFGS-B for local exploitation
        # Büche-Rastrigin has many local minima, so we restart from multiple points
        remaining = budget - calls_made
        if remaining <= 0 or time.time() >= deadline:
            return best_x, best_f

        # Generate restart points
        restarts = []
        
        # 1. Current best (if any)
        if best_x is not None:
            restarts.append(best_x.copy())
        
        # 2. Points near the best with varying perturbation scales
        if best_x is not None:
            for scale in [0.5, 1.0, 2.0, 3.0]:
                px = best_x + rng.standard_normal(dim) * scale
                px = np.clip(px, -5.0, 5.0)
                restarts.append(px)
        
        # 3. Random points in the box
        for _ in range(8):
            restarts.append(rng.uniform(-5, 5, dim))
        
        # 4. Grid-like points for structured coverage
        for i in range(4):
            pt = np.zeros(dim)
            for d in range(dim):
                pt[d] = -4.0 + (8.0 * i / 3.0)
            restarts.append(np.clip(pt + rng.standard_normal(dim) * 0.5, -5.0, 5.0))

        # Shuffle restarts
        rng.shuffle(restarts)

        # Run L-BFGS-B from each restart
        for rp in restarts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    make_eval,
                    np.clip(rp, -5.0, 5.0),
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': 50}
                )
                
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 3: Final polish around the best point
        if calls_made < budget and time.time() < deadline and best_x is not None:
            try:
                res = minimize(
                    make_eval,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxfun': 30, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Ensure best_x is valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = make_eval(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
