"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with Broyden refinement for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize at origin (often near the attractor for f6)
        x = np.zeros(dim)
        fx = func(x)
        calls_made = 1
        best_x = x.copy()
        best_f = fx

        def remaining():
            return budget - calls_made

        def time_ok():
            return time.time() < deadline

        # Phase 1: (1+1)-ES with 1/5 rule for coarse search
        sigma = 1.0
        success_count = 0
        window = 10
        window_count = 0

        while remaining() > 50 and time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            success = fy < fx
            if success:
                x = y
                fx = fy
                success_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_count -= 1

            window_count += 1

            if window_count >= window:
                success_rate = success_count / window
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.8
                success_count = 0
                window_count = 0

            if sigma < 1e-7:
                break

        # Phase 2: Broyden's method (BFGS-like) for refinement
        if remaining() > 20 and time_ok():
            try:
                new_best_x, new_best_f, calls_made = self._broyden_refine(
                    func, best_x, best_f, budget, calls_made, deadline, dim
                )
                best_x = new_best_x
                best_f = new_best_f
            except Exception:
                pass

        return best_x, best_f

    def _broyden_refine(self, func, x_init, fx_init, budget, calls_made_start, deadline, dim):
        """Broyden's method with finite-difference gradients."""
        h = 1e-5
        calls_made = calls_made_start
        best_f = fx_init
        best_x = x_init.copy()
        
        def compute_gradient(x):
            nonlocal calls_made
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                f_plus = func(np.clip(x_plus, -5, 5))
                f_minus = func(np.clip(x_minus, -5, 5))
                
                calls_made += 2
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad
        
        x = x_init.copy()
        fx = fx_init
        g0 = compute_gradient(x)
        H = np.eye(dim)
        
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        
        for iteration in range(50):
            if calls_made >= budget - 10:
                break
            if time.time() >= deadline:
                break
            
            d = -H @ g0
            
            step = alpha
            improved = False
            for _ in range(20):
                x_new = np.clip(x + step * d, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                
                if f_new < fx - c1 * step * np.dot(g0, d):
                    improved = True
                    break
                step *= rho
            
            if not improved:
                break
            
            x_new_clipped = np.clip(x + step * d, -5, 5)
            g_new = compute_gradient(x_new_clipped)
            
            s = x_new_clipped - x
            y = g_new - g0
            
            Hy = H @ y
            sTHs = s @ Hy
            if abs(sTHs) > 1e-12:
                H = H + np.outer(s - Hy, s @ H) / sTHs
            
            x = x_new_clipped
            fx = f_new
            g0 = g_new
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f, calls_made
# EVOLVE-BLOCK-END
