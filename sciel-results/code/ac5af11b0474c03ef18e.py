import numpy as np
import time
from scipy.optimize import minimize
import math
from scipy.optimize import approx_fprime


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: Initial sampling to find the basin
        # Include origin, structured points, and random samples
        n_structured = 8
        structured = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim),
            np.ones(self.dim) * -1.0,
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.ones(self.dim) * 0.1,
        ]
        n_random = 30
        random_pts = rng.uniform(-4, 4, (n_random, self.dim))

        all_pts = np.vstack([structured, random_pts])
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 5:
                break
            try:
                safe_func(p)
            except ValueError:
                break

        if best_x is None:
            # Fallback: return a default point
            return np.zeros(self.dim), 0.0

        # Phase 2: Custom derivative-free SQP-like BFGS with line search
        remaining = self.budget - call_count
        if remaining > 50 and time.time() < deadline - 10:
            try:
                # Initialize BFGS Hessian approximation
                H = np.eye(self.dim)
                x = best_x.copy()
                f_x = best_f
                
                # Compute initial gradient via finite differences
                eps = 1e-5
                grad = np.zeros(self.dim)
                for i in range(self.dim):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    f_plus = safe_func(x_plus)
                    f_minus = safe_func(x_minus)
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                
                # BFGS iteration loop
                for iteration in range(200):
                    if call_count >= self.budget or time.time() > deadline - 5:
                        break
                    
                    # Check if we're at a local minimum
                    if np.linalg.norm(grad) < 1e-10:
                        break
                    
                    # Newton step: p = -H @ grad
                    p = -H @ grad
                    p = p / (np.linalg.norm(p) + 1e-15)
                    
                    # Backtracking line search with box constraints
                    alpha = 1.0
                    c1 = 1e-4
                    for ls_iter in range(30):
                        x_trial = x + alpha * p
                        # Clip to box
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = safe_func(x_trial)
                        
                        if f_trial < f_x + c1 * alpha * np.dot(grad, p):
                            break
                        alpha *= 0.5
                    
                    if f_trial < f_x:
                        s = x_trial - x
                        f_x = f_trial
                        x = x_trial
                        
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x_trial.copy()
                        
                        # Compute new gradient
                        grad_new = np.zeros(self.dim)
                        for i in range(self.dim):
                            x_plus = x.copy()
                            x_plus[i] += eps
                            x_minus = x.copy()
                            x_minus[i] -= eps
                            f_plus = safe_func(x_plus)
                            f_minus = safe_func(x_minus)
                            grad_new[i] = (f_plus - f_minus) / (2 * eps)
                        
                        # BFGS update
                        y = grad_new - grad
                        sy = np.dot(s, y)
                        if sy > 1e-10:
                            H = H - np.outer(s, s) * (1.0 / sy) + np.outer(y, y) * (1.0 / np.dot(y, y))
                        grad = grad_new
                    else:
                        break
                        
            except Exception:
                pass

        # Phase 3: L-BFGS-B from current best with tighter tolerances
        remaining = self.budget - call_count
        if remaining > 30 and time.time() < deadline - 10:
            try:
                res_lbfgsb = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 300, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-12, 'ftol': 1e-15}
                )
                if res_lbfgsb.fun < best_f:
                    best_f = float(res_lbfgsb.fun)
                    best_x = np.array(res_lbfgsb.x, dtype=float)
            except Exception:
                pass

        # Phase 4: Final high-precision polish with Nelder-Mead
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 5:
            try:
                res_polish = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfun': remaining, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_polish.fun < best_f:
                    best_f = float(res_polish.fun)
                    best_x = np.array(res_polish.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
