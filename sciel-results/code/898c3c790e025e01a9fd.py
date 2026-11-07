"""Multi-start Nelder-Mead with Sobol initialization for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Multi-start Nelder-Mead with Sobol initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        history = []

        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            history.append((x, f_val))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Initial Exploration with Sobol ---
        n_sobol = 32
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points_unit = sampler.random(n_sobol)
            sobol_points = sobol_points_unit * 10.0 - 5.0
        except Exception:
            sobol_points = rng.uniform(-5.0, 5.0, (n_sobol, dim))

        for x in sobol_points:
            if eval_count >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if not history:
            return best_x, best_f

        # Sort history to find promising starting points
        sorted_hist = sorted(history, key=lambda item: item[1])
        
        # Queue of start points for local search
        # Start with the top 3 distinct points
        start_queue = []
        seen_starts = set()
        for x, f in sorted_hist[:5]:
            # Round to avoid duplicates in queue
            key = tuple(np.round(x, 4))
            if key not in seen_starts:
                start_queue.append(x)
                seen_starts.add(key)
        
        if not start_queue:
            start_queue.append(best_x)

        bounds = [(-5.0, 5.0)] * dim
        
        # --- Phase 2: Multi-start Local Optimization ---
        no_improve_count = 0
        max_no_improve = 50  # After 50 evals without improvement, add a new global start
        
        while eval_count < budget and time.time() < deadline:
            if not start_queue:
                # Generate a new random start or Sobol-like start
                # Use a larger step random walk from best or pure random
                if rng.random() < 0.5:
                    x_new = best_x + rng.normal(0, 2.0, dim)
                else:
                    x_new = rng.uniform(-5.0, 5.0, dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                start_queue.append(x_new)
                no_improve_count = 0 # Reset counter on new global exploration

            x_start = start_queue.pop(0)
            
            # Check time/budget before expensive local search
            if eval_count >= budget or time.time() > deadline:
                break

            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f - 1e-10:
                    # Improvement found
                    no_improve_count = 0
                    best_f = res.fun
                    best_x = res.x.copy()
                    
                    # Add perturbations of the new best to explore the basin
                    # 2 perturbations with small step
                    for _ in range(2):
                        x_pert = best_x + rng.normal(0, 0.5, dim)
                        x_pert = np.clip(x_pert, -5.0, 5.0)
                        start_queue.append(x_pert)
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Stagnation: force a new global start by clearing queue or adding aggressive random
                        # We already handle empty queue by adding random. 
                        # Let's just let the loop continue, but maybe add a more distant random point
                        x_far = rng.uniform(-5.0, 5.0, dim)
                        start_queue.append(x_far)
                        no_improve_count = 0 # Reset to avoid spamming

            except Exception:
                pass

        # --- Phase 3: Final Polish ---
        # Use L-BFGS-B for high precision on the best point found
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
