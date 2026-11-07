"""DFTR optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region method with diagonal quadratic surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point
        x = np.zeros(n)
        fx = func(x)
        f_best = fx
        x_best = x.copy()

        # Trust region radius
        rho = 1.0
        rho_min = 1e-4
        rho_max = 10.0

        calls_used = 1
        stall_count = 0

        # We need 10 calls for neighbors + 1 for step = 11 calls per iteration.
        # Plus potentially 1 for jump.
        while calls_used + 11 <= budget:
            # 1. Sample neighbors for model building
            # We use 2n points: x +/- rho/2 * e_j
            h = rho / 2.0
            f_plus = np.zeros(n)
            f_minus = np.zeros(n)
            
            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += h
                x_plus = np.clip(x_plus, -5, 5)
                f_plus[j] = func(x_plus)
                calls_used += 1
                
                x_minus = x.copy()
                x_minus[j] -= h
                x_minus = np.clip(x_minus, -5, 5)
                f_minus[j] = func(x_minus)
                calls_used += 1

            if calls_used >= budget:
                break

            # 2. Estimate gradient and diagonal Hessian
            # g_j = (f_plus - f_minus) / (2h)
            # D_jj = (f_plus + f_minus - 2fx) / (2h^2)
            
            g = (f_plus - f_minus) / (2 * h)
            d_diag = (f_plus + f_minus - 2 * fx) / (2 * h * h)
            
            # 3. Compute step
            # Model: f(d) = fx + sum(g_j d_j + D_jj d_j^2)
            # Optimal d_j = -g_j / (2 D_jj)
            d = np.zeros(n)
            for j in range(n):
                if abs(d_diag[j]) > 1e-10:
                    d[j] = -g[j] / (2 * d_diag[j])
                else:
                    # Linear term dominates, move against gradient
                    d[j] = -g[j]
            
            # Enforce trust region constraint ||d|| <= rho
            norm_d = np.linalg.norm(d)
            if norm_d > rho:
                d = d * (rho / norm_d)
            elif norm_d < 1e-12:
                # If step is tiny, try a small random step or increase rho?
                # For now, accept tiny step
                pass

            # 4. Evaluate new point
            x_new = x + d
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_used += 1
            
            # 5. Update Trust Region
            # Predicted change in f: pred_change = sum(g_j d_j + D_jj d_j^2)
            pred_change = np.sum(g * d) + np.sum(d_diag * d**2)
            act_improve = fx - f_new
            
            if act_improve > 1e-12:
                # Good step
                x = x_new
                fx = f_new
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                # Increase rho if improvement is good
                # Expected improvement is -pred_change
                if act_improve > 0.75 * (-pred_change):
                    rho = min(rho * 1.5, rho_max)
                else:
                    rho = min(rho * 1.1, rho_max)
            else:
                # Bad step
                rho = max(rho * 0.5, rho_min)
                stall_count += 1

            # Handle Stalls
            if stall_count > 5 and calls_used + 1 <= budget:
                # Random jump from best
                jump_scale = max(rho, 0.5)
                x = x_best + rng.normal(0, jump_scale, n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                rho = 1.0
                stall_count = 0
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
