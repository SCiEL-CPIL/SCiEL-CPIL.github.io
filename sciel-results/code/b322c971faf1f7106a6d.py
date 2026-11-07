"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive exploration for Katsuura."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin (330s limit, 45s buffer)
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def counted_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time exceeded")
            calls_made += 1
            return func(x)

        def nm_search(x0, maxiter, xatol=1e-8, fatol=1e-10):
            nonlocal best_x, best_f
            try:
                res = minimize(counted_func, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (RuntimeError, Exception):
                pass

        # Phase 1: Coarse exploration with multiple restarts
        # Katsuura has many local minima; we need diverse starting points
        n_restarts_phase1 = 8
        maxiter_phase1 = 15
        
        for i in range(n_restarts_phase1):
            if time.time() > deadline - 30:
                break
            if calls_made >= self.budget - 100:
                break
            
            # Use a mix of uniform and stratified starting points
            if i == 0:
                x0 = np.zeros(self.dim)
            elif i == 1:
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                # Stratified sampling for better coverage
                grid = np.linspace(-4.5, 4.5, 4)
                x0 = np.array([grid[rng.integers(0, 4)] + rng.uniform(-0.5, 0.5) for _ in range(self.dim)])
            
            nm_search(x0, maxiter_phase1)

        # Phase 2: Intermediate exploitation around best found
        if best_x is not None and time.time() < deadline - 20 and calls_made < self.budget - 50:
            remaining = self.budget - calls_made
            # Do a few more NM runs from perturbed versions of best
            n_phase2 = 4
            for j in range(n_phase2):
                if time.time() > deadline - 20:
                    break
                if calls_made >= self.budget - 30:
                    break
                
                # Perturb best solution
                perturbation_scale = 1.0 / (j + 1)
                x0 = best_x + perturbation_scale * rng.standard_normal(self.dim)
                x0 = np.clip(x0, -5, 5)
                
                maxiter_p2 = max(10, remaining // (n_phase2 - j))
                nm_search(x0, maxiter_p2, xatol=1e-9, fatol=1e-12)

        # Phase 3: Fine-tuning the best solution
        if best_x is not None and time.time() < deadline - 5 and calls_made < self.budget - 10:
            remaining = self.budget - calls_made
            maxiter_final = max(20, remaining // 8)
            nm_search(best_x, maxiter_final, xatol=1e-10, fatol=1e-14)

        # Phase 4: Final squeeze - try a few random points if budget remains
        if calls_made < self.budget - 5 and time.time() < deadline - 2:
            n_random = min(10, self.budget - calls_made)
            for _ in range(n_random):
                if time.time() > deadline - 2:
                    break
                x0 = best_x + 0.1 * rng.standard_normal(self.dim)
                x0 = np.clip(x0, -5, 5)
                try:
                    fx = counted_func(x0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x0.copy()
                except (RuntimeError, Exception):
                    break

        # Ensure at least one call was made
        if calls_made == 0:
            x0 = np.zeros(self.dim)
            best_f = func(x0)
            best_x = x0.copy()
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
