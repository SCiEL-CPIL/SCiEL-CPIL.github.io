"""Rewritten program using MODEL-AND-OPTIMIZE approach for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Model-based optimizer for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        bounds = [(-5.0, 5.0)] * self.dim

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # MODEL-AND-OPTIMIZE STRATEGY
        # The function is f(x) = ||x - x*||^2 + C
        # f(x) = x^T x - 2(x*)^T x + (||x*||^2 + C)
        # We can estimate x* by solving a linear system using N+1 points.
        
        N = self.dim
        # Create N+1 points to solve for N unknowns in x*
        # Use a symmetric design: center point and axis points at distance D
        D = 3.0  # Keep well within bounds [-5, 5]
        
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)
        
        # Evaluate these points
        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break
        
        # If we failed to get enough points, fallback to random search
        if len(values) < N + 1:
            # Fallback: Just do random search and local polish
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget: break
                x = rng.uniform(-5, 5, N)
                wrapped_func(x)
            if self.calls < self.budget:
                try:
                    minimize(wrapped_func, self.best_x, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 50, 'maxfun': self.budget - self.calls, 'gtol': 1e-8})
                except:
                    pass
            return (self.best_x, self.best_f)
        
        # Solve for x*
        # Equation: 2(x_i - x_j)^T x* = ||x_i||^2 - ||x_j||^2 - (f_i - f_j)
        # We use x_0 as the base for all differences.
        # For i from 1 to N:
        # 2(x_i - x_0)^T x* = ||x_i||^2 - ||x_0||^2 - (f_i - f_0)
        
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        x0 = points[0]
        f0 = values[0]
        
        for i in range(1, N + 1):
            xi = points[i]
            fi = values[i]
            
            diff = xi - x0
            norm_sq_diff = np.dot(xi, xi) - np.dot(x0, x0)
            val_diff = fi - f0
            
            A[i-1] = 2.0 * diff
            b[i-1] = norm_sq_diff - val_diff
        
        try:
            # Solve A x* = b
            x_est = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_est = np.zeros(N)
        
        # Clip the estimated center to the box
        x_start = np.clip(x_est, -5.0, 5.0)
        
        # Ensure we have evaluated the start point
        if not np.allclose(x_start, self.best_x):
            try:
                v = wrapped_func(x_start)
            except (ValueError, TimeoutError, Exception):
                pass
        
        # POLISH PHASE 1: L-BFGS-B for fast convergence
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 200, 
                        'maxfun': self.budget - self.calls,
                        'gtol': 1e-12
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass
        
        # POLISH PHASE 2: High-precision coordinate descent
        # For Sphere, coordinate descent is very effective near the optimum
        if self.calls < self.budget and time.time() < deadline:
            x_curr = self.best_x.copy()
            f_curr = self.best_f
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
            
            for step in step_sizes:
                if self.calls >= self.budget or time.time() > deadline:
                    break
                improved = True
                while improved and self.calls < self.budget and time.time() < deadline:
                    improved = False
                    for i in range(N):
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        # Try positive and negative steps
                        for direction in [1.0, -1.0]:
                            x_trial = x_curr.copy()
                            x_trial[i] += direction * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            if np.allclose(x_trial, x_curr):
                                continue
                            try:
                                f_trial = wrapped_func(x_trial)
                                if f_trial < f_curr:
                                    x_curr = x_trial
                                    f_curr = f_trial
                                    if f_curr < self.best_f:
                                        self.best_f = f_curr
                                        self.best_x = x_curr.copy()
                                    improved = True
                            except (ValueError, TimeoutError, Exception):
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
        
        # POLISH PHASE 3: Final L-BFGS-B with tighter tolerance if budget remains
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfun': self.budget - self.calls,
                        'gtol': 1e-14
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
