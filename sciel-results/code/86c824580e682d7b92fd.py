"""Rewritten program using MODEL-AND-OPTIMIZE with custom line search for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
        N = self.dim

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

        # PHASE 1: Estimate the center x* of the Sphere
        # f(x) = ||x - x*||^2 + c
        # f(x) = x^T x - 2 x*^T x + (||x*||^2 + c)
        # Let y = f(x) - x^T x. Then y = -2 x*^T x + K.
        # We use N+1 points: Origin and N axis points.
        
        D = 4.0
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)
        
        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break
        
        if len(values) < N + 1:
            # Fallback: Random search + L-BFGS-B if model fitting fails
            import scipy.optimize
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget: break
                x = rng.uniform(-5, 5, N)
                wrapped_func(x)
            if self.calls < self.budget:
                try:
                    scipy.optimize.minimize(wrapped_func, self.best_x, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 50, 'maxfun': self.budget - self.calls, 'gtol': 1e-8})
                except:
                    pass
            return (self.best_x, self.best_f)

        # Solve for x* and K
        # y_i = -2 x_i^T x* + K
        # A x* = b, where A is N x N, b is N x 1
        # Using x_0 as reference:
        # y_i - y_0 = -2 (x_i - x_0)^T x*
        # (y_0 - y_i) / 2 = (x_i - x_0)^T x*
        
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        x0 = points[0]
        y0 = values[0] - np.dot(x0, x0)
        
        for i in range(1, N + 1):
            xi = points[i]
            yi = values[i] - np.dot(xi, xi)
            
            diff = xi - x0
            A[i-1] = diff
            b[i-1] = (y0 - yi) / 2.0
        
        try:
            x_est = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_est = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Clip estimated center to bounds
        x_start = np.clip(x_est, -5.0, 5.0)
        
        # Evaluate start point
        try:
            v = wrapped_func(x_start)
        except (ValueError, TimeoutError, Exception):
            pass

        # PHASE 2: Polish using custom line search
        # For Sphere, Hessian is 2I. Gradient is 2(x - x*).
        # Optimal step size is 0.5.
        # We use a line search to be robust against model errors.
        
        x_cur = x_start.copy()
        f_cur = self.best_f
        
        # If we are already at the estimated center, we might be done.
        # But let's do a few polish steps to ensure convergence.
        
        max_polish_steps = 50
        step_count = 0
        
        while self.calls < self.budget and time.time() < deadline and step_count < max_polish_steps:
            # Check convergence
            dist_to_center = np.linalg.norm(x_cur - x_est)
            if dist_to_center < 1e-8:
                break
            
            # Model gradient
            g = 2.0 * (x_cur - x_est)
            # Direction
            d = -g
            
            # Line Search
            # We expect alpha ~ 0.5.
            # Use a simple bracketing or parabolic interpolation.
            # Let's try 3 points around 0.5 to fit a parabola.
            alpha_1 = 0.25
            alpha_2 = 0.50
            alpha_3 = 0.75
            
            x_1 = x_cur + alpha_1 * d
            x_2 = x_cur + alpha_2 * d
            x_3 = x_cur + alpha_3 * d
            
            # Clip points to bounds before evaluation? 
            # The wrapped_func clips, but for line search math, we should be careful.
            # If bounds are active, the line search might fail.
            # For now, rely on wrapped_func clipping.
            
            try:
                f_1 = wrapped_func(x_1)
                f_2 = wrapped_func(x_2)
                f_3 = wrapped_func(x_3)
            except (ValueError, TimeoutError, Exception):
                break
            
            # Fit parabola to (alpha_1, f_1), (alpha_2, f_2), (alpha_3, f_3)
            # f(a) = A a^2 + B a + C
            # We want to find a* = -B / (2A)
            
            # Using Lagrange interpolation or solving linear system
            # Let's use a simpler method: 
            # If f_2 < f_1 and f_2 < f_3, minimum is near 0.5.
            # We can use the vertex formula for 3 points.
            
            # Solve for A, B, C:
            # A a1^2 + B a1 + C = f1
            # A a2^2 + B a2 + C = f2
            # A a3^2 + B a3 + C = f3
            
            # Subtracting to eliminate C:
            # A(a1^2 - a2^2) + B(a1 - a2) = f1 - f2
            # A(a3^2 - a2^2) + B(a3 - a2) = f3 - f2
            
            # Let h1 = a1 - a2, h3 = a3 - a2
            # A(a1+a2)h1 + B h1 = f1 - f2
            # A(a3+a2)h3 + B h3 = f3 - f2
            
            # This is a 2x2 system for A and B.
            
            h1 = alpha_1 - alpha_2
            h3 = alpha_3 - alpha_2
            s1 = alpha_1 + alpha_2
            s3 = alpha_3 + alpha_2
            df1 = f_1 - f_2
            df3 = f_3 - f_2
            
            # Matrix:
            # [s1*h1, h1] [A]   [df1]
            # [s3*h3, h3] [B] = [df3]
            
            det = (s1 * h1) * h3 - (s3 * h3) * h1
            if abs(det) < 1e-10:
                # Degenerate, just take alpha_2
                alpha_best = alpha_2
            else:
                A_val = (df1 * h3 - df3 * h1) / det
                B_val = (df1 - A_val * s1 * h1) / h1 if abs(h1) > 1e-10 else 0
                
                if abs(A_val) < 1e-10:
                    alpha_best = alpha_2
                else:
                    alpha_best = -B_val / (2 * A_val)
                    
                    # Clip alpha to reasonable range
                    alpha_best = np.clip(alpha_best, 0.0, 1.0)
            
            # Move to best alpha
            x_new = x_cur + alpha_best * d
            
            try:
                f_new = wrapped_func(x_new)
            except (ValueError, TimeoutError, Exception):
                break
            
            if f_new < f_cur:
                x_cur = x_new
                f_cur = f_new
                step_count += 1
            else:
                # If we didn't improve, try a smaller step or break
                # For a convex function, this shouldn't happen often if alpha is good
                break
        
        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
