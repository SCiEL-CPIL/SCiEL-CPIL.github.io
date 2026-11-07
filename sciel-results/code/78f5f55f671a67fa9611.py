"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Hybrid Derivative-Free BFGS + Nelder-Mead for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_fx
            # Ensure x is a finite 1D array of size n
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            val = func(x)
            calls += 1
            if np.isfinite(val) and val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Initial Exploration
        # Use a mix of structured and random points to quickly identify a good basin.
        # Rosenbrock minimum is typically at (1, ..., 1) in standard form, but shifted/rotated here.
        # We sample around 0, 1, and random points.
        
        initial_candidates = [
            np.zeros(n),
            np.ones(n) * 1.0,
            np.ones(n) * 0.5,
            np.ones(n) * -1.0,
            np.ones(n) * -0.5,
            np.ones(n) * 2.0,
            np.ones(n) * -2.0
        ]
        
        # Add random points
        n_random = 15
        rand_points = rng.uniform(-4, 4, (n_random, n))
        initial_candidates.extend(rand_points)
        
        for x in initial_candidates:
            if time.time() > deadline or calls >= budget:
                break
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: Derivative-Free BFGS Local Search
        # We estimate gradients using central finite differences.
        # Cost per step: 2*n evaluations.
        # We maintain an inverse Hessian approximation H.
        
        def df_bfgs(start_x, max_iter, h=1e-4):
            nonlocal calls, best_x, best_fx
            x = start_x.copy()
            fx = safe_eval(x)
            if not np.isfinite(fx):
                return x, fx
            
            H = np.eye(n)  # Inverse Hessian approximation
            g_prev = None
            x_prev = None
            
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Estimate gradient using central differences
                g = np.zeros(n)
                for i in range(n):
                    if calls >= budget: break
                    x_plus = x.copy()
                    x_plus[i] += h
                    x_minus = x.copy()
                    x_minus[i] -= h
                    # Clip to bounds to be safe, though func clips
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)
                    f_plus = safe_eval(x_plus)
                    f_minus = safe_eval(x_minus)
                    if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                        g[i] = 0.0
                    else:
                        g[i] = (f_plus - f_minus) / (2 * h)
                
                if calls >= budget:
                    break
                
                # Check gradient norm for convergence
                if np.linalg.norm(g) < 1e-6:
                    break
                
                # Search direction: d = -H @ g
                d = -H @ g
                
                # Line search with backtracking
                alpha = 1.0
                c1 = 1e-4  # Armijo condition
                max_alpha = 10.0
                min_alpha = 1e-6
                step_accepted = False
                best_alpha = 0.0
                best_f_alpha = fx
                
                while alpha >= min_alpha:
                    x_new = x + alpha * d
                    x_new = np.clip(x_new, -5, 5)
                    f_new = safe_eval(x_new)
                    
                    if np.isfinite(f_new):
                        # Armijo condition: f_new <= fx + c1 * alpha * g^T d
                        if f_new <= fx + c1 * alpha * (g @ d):
                            step_accepted = True
                            best_alpha = alpha
                            best_f_alpha = f_new
                            break
                        if f_new < best_f_alpha:
                            best_alpha = alpha
                            best_f_alpha = f_new
                    
                    alpha *= 0.5
                
                if not step_accepted:
                    # If no step accepted, try the best alpha found anyway or break
                    if best_alpha > 0 and np.isfinite(best_f_alpha):
                        x = x + best_alpha * d
                        x = np.clip(x, -5, 5)
                        fx = best_f_alpha
                    else:
                        break
                
                # Update BFGS
                if step_accepted:
                    x_new = x + best_alpha * d
                    x_new = np.clip(x_new, -5, 5)
                    f_new = best_f_alpha
                    s = x_new - x
                    y = None # We need to re-estimate gradient at new point for BFGS update
                    # To save calls, we can skip the BFGS update if we are close to budget
                    # or estimate gradient again. Let's estimate gradient again.
                    if calls + 2*n > budget:
                        break
                    
                    g_new = np.zeros(n)
                    for i in range(n):
                        if calls >= budget: break
                        x_plus = x_new.copy()
                        x_plus[i] += h
                        x_minus = x_new.copy()
                        x_minus[i] -= h
                        x_plus = np.clip(x_plus, -5, 5)
                        x_minus = np.clip(x_minus, -5, 5)
                        f_plus = safe_eval(x_plus)
                        f_minus = safe_eval(x_minus)
                        if np.isfinite(f_plus) and np.isfinite(f_minus):
                            g_new[i] = (f_plus - f_minus) / (2 * h)
                        else:
                            g_new[i] = 0.0
                    
                    y = g_new - g
                    sy = s @ y
                    if sy > 1e-10:
                        # BFGS update for inverse Hessian H
                        rho = 1.0 / sy
                        H = H + rho * (np.outer(s, s) - np.outer(s, H @ y) - np.outer(H @ y, s) + np.outer(H @ y, H @ y) * rho * (y @ H @ y))
                        # Ensure symmetry and positive definite-ish
                        H = (H + H.T) / 2.0
                        
                    x = x_new
                    fx = f_new
                    g = g_new
                else:
                    break

            return x, fx

        # Run BFGS from best_x
        # Max iterations: (budget - calls) // (4*n) approx, since we use 2n for grad, 1 for step, 2n for new grad
        # Let's be conservative: 20 iterations
        max_bfgs_iter = 20
        if calls < budget:
            bfgs_x, bfgs_fx = df_bfgs(best_x, max_bfgs_iter)
            if np.isfinite(bfgs_fx) and bfgs_fx < best_fx:
                best_x = bfgs_x
                best_fx = bfgs_fx

        # Phase 3: Local Polish with Nelder-Mead
        # If we have budget left, use NM for fine-tuning
        if best_x is not None and time.time() < deadline and calls < budget:
            def nm_wrapper(x):
                nonlocal calls, best_x, best_fx
                if time.time() > deadline:
                    raise ValueError("Time limit exceeded")
                if calls >= budget:
                    raise ValueError("Budget exceeded")
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if np.isfinite(val) and val < best_fx:
                    best_fx = val
                    best_x = x_clipped.copy()
                return val

            try:
                res = scipy.optimize.minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(budget - calls, 100),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            if calls < budget:
                safe_eval(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
