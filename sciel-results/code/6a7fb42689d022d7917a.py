"""Multi-start local optimizer with Sobol exploration for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Hybrid Sobol + Nelder-Mead optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Store evaluated points for multi-start
        all_points = []
        all_values = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            all_points.append(x.copy())
            all_values.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Sobol Space-Filling Exploration ---
        n_sobol = min(80, max(32, budget // 5))
        
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points_unit = sampler.random(n_sobol)
            sobol_points = sobol_points_unit * 10.0 - 5.0
            
            for x in sobol_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                wrapped_func(x)
        except Exception:
            for _ in range(n_sobol):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x)

        # --- Phase 2: Multi-start Local Refinement from Top Candidates ---
        # Get top K candidates from exploration
        k_candidates = min(5, len(all_values))
        if k_candidates > 0:
            sorted_idx = np.argsort(all_values)[:k_candidates]
            top_candidates = [all_points[i] for i in sorted_idx]
        else:
            top_candidates = [best_x]

        bounds = [(-5.0, 5.0)] * dim
        last_improvement_eval = eval_count
        stagnation_threshold = 15

        # Refine each top candidate with Nelder-Mead
        for start_x in top_candidates:
            if eval_count >= budget or time.time() > deadline:
                break
            try:
                res = minimize(
                    wrapped_func,
                    start_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
            except Exception:
                pass

        # --- Phase 3: Main Loop with Adaptive Stagnation ---
        # Use a population-based approach: maintain a set of active points
        # and periodically add new random points when stagnating
        
        active_starts = [best_x.copy()]
        last_best_f = best_f
        no_improve_count = 0
        adapt_radius = 2.0  # Exploration radius, adapts over time

        while eval_count < budget and time.time() < deadline:
            # Check stagnation: if no improvement in last 15 evals
            if best_f < last_best_f - 1e-10:
                no_improve_count = 0
                last_best_f = best_f
            else:
                no_improve_count += 1

            # If stagnating, add a new random start point
            if no_improve_count >= stagnation_threshold:
                no_improve_count = 0
                # Choose between: pure random, or perturbation of best
                if rng.random() < 0.4:
                    x_new = rng.uniform(-5.0, 5.0, dim)
                else:
                    x_new = best_x + rng.normal(0, adapt_radius, dim)
                    x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = wrapped_func(x_new)
                active_starts.append(x_new.copy())
                
                # Keep at most 3 active starts
                if len(active_starts) > 3:
                    active_starts.pop(0)
                
                # Shrink exploration radius as we converge
                if best_f < 0:
                    adapt_radius = max(0.5, adapt_radius * 0.8)
            else:
                # Exploit: pick the best active start and refine
                # Use the most recent active start (likely closest to improvement)
                start_x = active_starts[-1] if active_starts else best_x
                
                try:
                    res = minimize(
                        wrapped_func,
                        start_x.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 40, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                    )
                except Exception:
                    pass

        # --- Phase 4: Final High-Precision Polish ---
        # Use tighter tolerances and more iterations on the best point
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
            except Exception:
                pass

        # Final local sampling around best with adaptive step size
        remaining = budget - eval_count
        if remaining > 0 and time.time() < deadline:
            # Adaptive step: smaller if we're close to convergence
            step_size = max(0.01, 0.5 * np.exp(-0.1 * (budget - eval_count) / 10.0))
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                x = best_x + rng.normal(0, step_size, dim)
                x = np.clip(x, -5.0, 5.0)
                wrapped_func(x)

        return best_x, best_f

# EVOLVE-BLOCK-END
