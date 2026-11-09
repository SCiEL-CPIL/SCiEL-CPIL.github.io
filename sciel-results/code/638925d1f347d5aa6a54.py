"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES-like optimizer with multi-start for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            calls += 1
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Multi-start with local refinement
        # Weierstrass is multi-modal; try several starting points
        rng = np.random.default_rng(42)
        
        # Try origin first (global min often near center after shift)
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        # Try a few random starts
        for _ in range(4):
            if calls >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-2, 2, dim)
            f0 = evaluate(x0)
        
        # Phase 2: Use Nelder-Mead for local refinement from best point
        if calls < budget and time.time() < deadline:
            def wrapper(x):
                nonlocal calls
                if time.time() > deadline or calls >= budget:
                    return np.inf
                return evaluate(x)
            
            # Nelder-Mead from best point
            res = minimize(wrapper, best_x, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
        
        # Phase 3: If calls remain, do another round of local search from best
        if calls < budget - 10 and time.time() < deadline:
            def wrapper2(x):
                nonlocal calls
                if time.time() > deadline or calls >= budget:
                    return np.inf
                return evaluate(x)
            
            # Perturb and refine
            for _ in range(3):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                x_perturb = best_x + rng.uniform(-0.5, 0.5, dim)
                res = minimize(wrapper2, x_perturb, method='Nelder-Mead',
                             options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
        
        return best_x, best_f
# EVOLVE-BLOCK-END
