"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer with Rastrigin-aware initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(12345)

        # Phase 1: Smart initialization - sample near integers (Rastrigin peaks)
        # and random points to find good basins
        n_init = min(30, max_calls // 15)
        init_points = []

        # Add points near integer coordinates (Rastrigin has local minima near integers)
        for i in range(n_init // 2):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = np.round(rng.uniform(-4, 4, dim))
            x = np.clip(x, -5, 5)
            f = counted_func(x)
            init_points.append((f, x.copy()))

        # Add random points
        for i in range(n_init - len(init_points)):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)
            init_points.append((f, x.copy()))

        # Also try origin
        if call_count < max_calls:
            x_origin = np.zeros(dim)
            f_origin = counted_func(x_origin)
            init_points.append((f_origin, x_origin.copy()))

        init_points.sort(key=lambda t: t[0])

        # Phase 2: Multi-start L-BFGS-B from top initialization points
        # L-BFGS-B is good for smooth functions and handles bounds well
        n_starts = min(10, len(init_points))
        
        for f_init, x_init in init_points[:n_starts]:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            remaining = max_calls - call_count
            if remaining < 20:
                break

            try:
                result = minimize(
                    counted_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
                    
            except Exception:
                pass

        # Phase 3: If budget remains, do Nelder-Mead refinement on best point
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 30:
                try:
                    nm_res = minimize(
                        counted_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xtol': 1e-10,
                            'ftol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if nm_res.fun < best_f:
                        best_f = nm_res.fun
                        best_x = np.clip(nm_res.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 4: Final fine-tuning with coordinate descent if budget remains
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 15:
                x = best_x.copy()
                f_curr = best_f
                
                # Try small perturbations in each dimension
                for step in [0.01, 0.005, 0.001]:
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    improved = True
                    while improved and call_count < max_calls and time.time() < deadline:
                        improved = False
                        for d in range(dim):
                            for direction in [-1, 1]:
                                if call_count >= max_calls or time.time() > deadline:
                                    break
                                x_trial = x.copy()
                                x_trial[d] += direction * step
                                if x_trial[d] < -5 or x_trial[d] > 5:
                                    continue
                                f_trial = counted_func(x_trial)
                                if f_trial < f_curr:
                                    x = x_trial
                                    f_curr = f_trial
                                    improved = True
                                    if f_curr < best_f:
                                        best_f = f_curr
                                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
