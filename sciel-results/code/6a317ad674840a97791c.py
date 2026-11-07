"""Rewritten program using LHS + custom FD-BFGS + L-BFGS-B hybrid strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    """Hybrid LHS, custom FD-BFGS, and L-BFGS-B optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Deadline guard: keep 10s margin (300s limit)
        deadline = start_time + 290

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                if len(x) > dim:
                    x = x[:dim]
                else:
                    x = np.pad(x, (0, dim - len(x)))
            x = np.clip(x, -5.0, 5.0)
            
            call_count += 1
            f_val = func(x)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # 1. Global Search: Latin Hypercube Sampling
        n_global = 100
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_global)
        # Scale from [0, 1] to [-4, 4]
        lhs_points = lhs_points * 8.0 - 4.0

        global_results = []
        for x in lhs_points:
            if time.time() > deadline:
                break
            if call_count >= budget - 100:
                break
            f_val = wrapped_func(x)
            global_results.append((f_val, x))

        if not global_results:
            x0 = np.zeros(dim)
            wrapped_func(x0)
            return best_x, best_f

        global_results.sort(key=lambda item: item[0])
        
        # 2. Local Search: Custom FD-BFGS on top candidate
        def fd_bfgs(x0, max_iters=50, h=1e-6):
            """Custom finite-difference gradient with BFGS quasi-Newton updates."""
            nonlocal call_count, best_f, best_x
            x = x0.copy()
            f_curr = wrapped_func(x)
            
            # Initialize Hessian inverse approximation as identity
            H = np.eye(dim)
            g_prev = None
            x_prev = None
            f_prev = None
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if call_count >= budget - 10:
                    break
                
                # Finite difference gradient estimation
                grad = np.zeros(dim)
                for i in range(dim):
                    if time.time() > deadline or call_count >= budget - 10:
                        break
                    x_plus = x.copy()
                    x_plus[i] += h
                    f_plus = wrapped_func(x_plus)
                    
                    if time.time() > deadline or call_count >= budget - 10:
                        break
                    x_minus = x.copy()
                    x_minus[i] -= h
                    f_minus = wrapped_func(x_minus)
                    
                    grad[i] = (f_plus - f_minus) / (2 * h)
                
                # Check for convergence
                if np.linalg.norm(grad) < 1e-8:
                    break
                
                # BFGS update
                if g_prev is not None and x_prev is not None:
                    s = x - x_prev
                    y = grad - g_prev
                    sy = s.dot(y)
                    if sy > 1e-12:
                        Hy = H.dot(y)
                        H = H - (np.outer(s, s) * (y.dot(Hy) / sy + 1.0) / sy)
                        H = H + (np.outer(s, Hy) + np.outer(Hy, s)) / sy
                
                # Newton step with line search
                d = -H.dot(grad)
                
                # Backtracking line search
                alpha = 1.0
                c1 = 1e-4
                for ls_iter in range(20):
                    if time.time() > deadline or call_count >= budget - 10:
                        break
                    x_new = x + alpha * d
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = wrapped_func(x_new)
                    
                    if f_new < f_curr - c1 * alpha * grad.dot(d):
                        break
                    alpha *= 0.5
                
                # Update
                x_prev = x.copy()
                f_prev = f_curr
                g_prev = grad.copy()
                x = x_new
                f_curr = f_new
                
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()
            
            return x, f_curr

        top_k = 2
        starts = [p for _, p in global_results[:top_k]]
        
        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if call_count >= budget - 20:
                break
            
            max_iters_local = 50 if i == 0 else 30
            fd_bfgs(x0, max_iters=max_iters_local, h=1e-6)

        # 3. Polish: L-BFGS-B on the best point found
        if time.time() < deadline and call_count < budget - 5:
            try:
                bounds = [(-5.0, 5.0)] * dim
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
