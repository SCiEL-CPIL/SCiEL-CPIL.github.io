import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Hybrid optimizer: structured multi-start + L-BFGS-B + adaptive coordinate descent.
    Combines global exploration with local refinement exploiting separable structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(123)

        # Phase 1: Initial structured sampling to find good starting points
        initial_points = []
        initial_points.append(np.zeros(dim))
        for _ in range(8):
            initial_points.append(rng.uniform(-4.5, 4.5, dim))
        for _ in range(4):
            initial_points.append(rng.choice([-4.0, 0.0, 4.0], dim))

        for pt in initial_points:
            if eval_count >= budget - 50 or time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: Multi-start L-BFGS-B from diverse candidates
        start_points = [best_x.copy()]

        for scale in [1.0, 0.5, 0.25]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)

        for _ in range(6):
            start_points.append(rng.uniform(-4.5, 4.5, dim))

        for _ in range(3):
            start_points.append(rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0], dim))

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)

        for sp in unique_starts:
            if eval_count >= budget - 10 or time.time() > deadline:
                break
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 15, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Adaptive coordinate descent with quadratic line-search
        # This exploits f22's separable structure near peaks
        if eval_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            for iteration in range(50):
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    if eval_count >= budget - 5 or time.time() > deadline:
                        break
                    
                    # Try moving along axis d
                    step = 0.1
                    for _ in range(10):
                        if eval_count >= budget - 5 or time.time() > deadline:
                            break
                        
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x[d] + step, -5.0, 5.0)
                        f_trial = wrapped_func(x_trial)
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            step *= 1.2  # increase step if improving
                        else:
                            step *= 0.5  # decrease step if not improving
                            break
                    
                    # Also try negative direction
                    if eval_count < budget - 5 and time.time() < deadline:
                        step = 0.1
                        for _ in range(10):
                            if eval_count >= budget - 5 or time.time() > deadline:
                                break
                            
                            x_trial = x.copy()
                            x_trial[d] = np.clip(x[d] - step, -5.0, 5.0)
                            f_trial = wrapped_func(x_trial)
                            
                            if f_trial < fx:
                                x = x_trial
                                fx = f_trial
                                improved = True
                                step *= 1.2
                            else:
                                step *= 0.5
                                break
                
                if not improved:
                    break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 4: Final refinement with Nelder-Mead if budget remains
        if eval_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Fill remaining budget with evaluations around best if needed
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
