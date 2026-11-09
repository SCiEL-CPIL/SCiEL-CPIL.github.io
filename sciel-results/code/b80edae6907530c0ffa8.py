"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, plus BFGS-like local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 2.0
        calls_used = 1

        # Phase 1: (1+1)-ES with adaptive sigma
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.5
            else:
                sigma *= 0.87

        # Phase 2: BFGS-like local polish around the best point found
        # Use finite-difference gradients and quasi-Newton updates
        x_best = best_x.copy()
        fx_best = best_fx

        # Estimate gradient using central differences
        def estimate_gradient(x, f, h=1e-4):
            g = np.zeros(self.dim)
            for i in range(self.dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] = min(5.0, xp[i] + h)
                xm[i] = max(-5.0, xm[i] - h)
                g[i] = (f(xp) - f(xm)) / (xp[i] - xm[i])
            return g

        # Check if we have enough budget left for polish
        if calls_used < self.budget - 2 * self.dim:
            # BFGS initialization
            H = np.eye(self.dim)  # inverse Hessian approximation
            x_bfgs = x_best.copy()
            fx_bfgs = fx_best
            g = estimate_gradient(x_bfgs, func)
            calls_used += 2 * self.dim

            for it in range(min(50, self.budget - calls_used - 2 * self.dim)):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break

                # Search direction
                d = -H @ g
                # Line search
                alpha = 1.0
                for ls in range(20):
                    if time.time() > deadline:
                        break
                    x_new = np.clip(x_bfgs + alpha * d, -5, 5)
                    fx_new = func(x_new)
                    calls_used += 1
                    if fx_new < fx_bfgs:
                        break
                    alpha *= 0.5
                else:
                    break

                g_new = estimate_gradient(x_new, func)
                calls_used += 2 * self.dim

                if fx_new < fx_bfgs:
                    s = x_new - x_bfgs
                    y_vec = g_new - g
                    sy = s @ y_vec
                    if sy > 1e-10:
                        # BFGS update
                        Hy = H @ y_vec
                        H = H + (sy + y_vec @ Hy) / (sy * sy) * np.outer(s, s) - np.outer(Hy, s) - np.outer(s, Hy)
                        x_bfgs = x_new
                        fx_bfgs = fx_new
                        g = g_new
                        if fx_bfgs < best_fx:
                            best_x, best_fx = x_bfgs.copy(), fx_bfgs
                    else:
                        x_bfgs = x_new
                        fx_bfgs = fx_new
                        g = g_new
                        if fx_bfgs < best_fx:
                            best_x, best_fx = x_bfgs.copy(), fx_bfgs
                else:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
