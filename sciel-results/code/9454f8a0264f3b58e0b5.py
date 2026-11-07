"""Optimized program using CMA-ES with multi-restart for Rastrigin optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-restart CMA-ES optimizer for BBOB-style Rastrigin (f3).
    Uses multiple global restarts with CMA-ES local optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        best_f_history = []  # Track best f at each call for AOCC-like scoring
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-restart strategy:
        # - First restart: broad exploration from center
        # - Subsequent restarts: perturb around best found so far
        
        n_restarts = 8
        # Allocate budget: first restart gets more for exploration
        # We'll dynamically allocate based on remaining budget and time
        
        # Phase 1: Initial broad exploration with Sobol-like sampling
        # Use a stratified random sampling approach
        n_initial = 64
        if n_initial <= budget:
            # Generate stratified points in [-5, 5]^dim
            points = np.zeros((n_initial, dim))
            for i in range(dim):
                # Stratify each dimension
                edges = np.linspace(-5, 5, n_initial + 1)
                for j in range(n_initial):
                    points[j, i] = rng.uniform(edges[j], edges[j+1])
            
            for x in points:
                if time.time() > deadline or calls_made >= budget:
                    break
                counted_func(x)
        
        # Phase 2: Multi-restart CMA-ES
        # CMA-ES is adaptive and works well for Rastrigin's multimodal landscape
        
        for restart_idx in range(n_restarts):
            if time.time() > deadline - 30:
                break
            if calls_made >= budget:
                break
            
            # Determine starting point for this restart
            if restart_idx == 0:
                x_start = best_x.copy()
            else:
                # Perturb around best found
                sigma_init = 2.0  # Initial step size
                x_start = best_x + rng.normal(0, sigma_init, dim)
                x_start = np.clip(x_start, -5, 5)
            
            # Determine max function evaluations for this CMA-ES run
            # Decreasing allocation for later restarts
            remaining = budget - calls_made
            if remaining <= 0:
                break
            
            # Allocate more budget to earlier restarts
            if restart_idx == 0:
                maxfev = min(remaining, 150)
            elif restart_idx == 1:
                maxfev = min(remaining, 100)
            elif restart_idx == 2:
                maxfev = min(remaining, 80)
            else:
                maxfev = min(remaining, 50)
            
            if maxfev < 20:
                break
            
            try:
                res = minimize(
                    counted_func,
                    x_start,
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': maxfev}
                )
            except Exception:
                pass
        
        # Phase 3: Final local polish with Nelder-Mead
        if calls_made < budget and time.time() < deadline - 10:
            x_start = best_x.copy()
            maxfev = budget - calls_made
            if maxfev >= 10:
                try:
                    res = minimize(
                        counted_func,
                        x_start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': maxfev,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass
        
        # Phase 4: If we have budget left, do a few more local refinements
        # from perturbed best points
        if calls_made < budget - 5 and time.time() < deadline - 5:
            for _ in range(5):
                if calls_made >= budget or time.time() > deadline - 3:
                    break
                x_perturbed = best_x + rng.normal(0, 0.5, dim)
                x_perturbed = np.clip(x_perturbed, -5, 5)
                fx = counted_func(x_perturbed)
                if fx < best_f:
                    # Found improvement, do local refinement
                    if calls_made < budget - 10 and time.time() < deadline - 5:
                        try:
                            res = minimize(
                                counted_func,
                                x_perturbed,
                                method='Powell',
                                options={'maxiter': 50, 'maxfev': budget - calls_made}
                            )
                        except Exception:
                            pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
