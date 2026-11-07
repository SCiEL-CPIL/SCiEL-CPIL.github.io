"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer for BBOB f9 (rotated Rosenbrock).

    Strategy:
    1. Central finite-difference gradient estimation.
    2. Variable-step gradient descent with backtracking line search.
    3. Periodic restarts from multiple starting points if stuck.
    4. Time-aware budget management to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _estimate_gradient(self, func, x, h=1e-5):
        """Central finite difference gradient."""
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            # Clip to box to ensure valid evaluation
            xp = np.clip(xp, -5, 5)
            xm = np.clip(xm, -5, 5)
            grad[i] = (func(xp) - func(xm)) / (2 * h)
        return grad

    def _backtracking_line_search(self, func, x, fx, grad, alpha=1.0, c1=1e-4, rho=0.5):
        """Backtracking line search to find a step size that satisfies Armijo condition."""
        n = self.dim
        for _ in range(60):
            y = np.clip(x - alpha * grad, -5, 5)
            fy = func(y)
            # Armijo condition: f(y) <= f(x) + c1 * alpha * ||grad||^2
            if fy <= fx + c1 * alpha * np.dot(grad, grad):
                return y, fy, alpha
            alpha *= rho
        # If line search fails, take the smallest step
        y = np.clip(x - alpha * grad, -5, 5)
        fy = func(y)
        return y, fy, alpha

    def _gradient_descent(self, func, x0, fx0, max_iters, deadline):
        """Run gradient descent with backtracking line search."""
        x = x0.copy()
        fx = fx0
        alpha = 0.01
        no_improve_count = 0
        best_x = x.copy()
        best_fx = fx

        for i in range(max_iters):
            if time.time() > deadline:
                break
            # Estimate gradient
            grad = self._estimate_gradient(func, x)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-8:
                break
            
            # Backtracking line search
            y, fy, alpha_used = self._backtracking_line_search(func, x, fx, grad, alpha)
            
            if fy < fx:
                x = y
                fx = fy
                alpha = min(alpha_used * 1.5, 1.0)  # Increase step if it worked
                no_improve_count = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # No improvement; reduce step size
                alpha *= 0.5
                no_improve_count += 1
                if no_improve_count > 20:
                    break
        
        return best_x, best_fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        
        # Initial point: try center first, then random
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Phase 1: Gradient descent from center
        max_iters_phase1 = 100
        x, fx = self._gradient_descent(func, x, fx, max_iters_phase1, deadline)
        if fx < best_fx:
            best_fx = fx
            best_x = x.copy()
        
        # Phase 2: Multiple restarts from random points
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random starting point
            x_r = rng.uniform(-5, 5, self.dim)
            fx_r = func(x_r)
            
            # If this point is already good, use it
            if fx_r < best_fx:
                best_fx = fx_r
                best_x = x_r.copy()
            
            # Run gradient descent from this point
            max_iters = 80
            x_g, fx_g = self._gradient_descent(func, x_r, fx_r, max_iters, deadline)
            if fx_g < best_fx:
                best_fx = fx_g
                best_x = x_g.copy()
        
        # Phase 3: Local refinement around best point with smaller steps
        if time.time() < deadline - 30:
            x_ref = best_x.copy()
            fx_ref = best_fx
            # Use smaller finite difference step for refinement
            for _ in range(50):
                if time.time() > deadline:
                    break
                h = 1e-6
                grad = np.zeros(self.dim)
                for i in range(self.dim):
                    xp = x_ref.copy()
                    xm = x_ref.copy()
                    xp[i] += h
                    xm[i] -= h
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    grad[i] = (func(xp) - func(xm)) / (2 * h)
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-9:
                    break
                
                # Small step
                alpha = 0.001
                y = np.clip(x_ref - alpha * grad / grad_norm * min(grad_norm, 1.0), -5, 5)
                fy = func(y)
                if fy < fx_ref:
                    x_ref = y
                    fx_ref = fy
                    if fx_ref < best_fx:
                        best_fx = fx_ref
                        best_x = x_ref.copy()
                else:
                    break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
