"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
        """Exact affine model fitting for BBOB f5 (Linear slope)."""

        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290.0

            best_x = None
            best_f = np.inf

            def update_best(x, f):
                nonlocal best_x, best_f
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            dim = self.dim
            budget = self.budget
            calls_used = 0

            # Phase 1: Fit exact affine model f(x) = w^T x + b
            # Evaluate at origin and at unit basis vectors scaled to reach boundary
            
            x0 = np.zeros(dim)
            f0 = func(x0)
            calls_used += 1
            update_best(x0, f0)

            if time.time() > deadline:
                return best_x, best_f

            step = 5.0
            b_vec = np.zeros(dim)

            for i in range(dim):
                if time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] = step
                fi = func(xi)
                calls_used += 1
                update_best(xi, fi)
                b_vec[i] = fi - f0

            # Solve for gradient: w_i = (fi - f0) / step
            w = b_vec / step
            b = f0

            # Compute the optimal vertex of [-5, 5]^d for the linear model
            x_opt = np.zeros(dim)
            for i in range(dim):
                if w[i] > 0:
                    x_opt[i] = -5.0
                elif w[i] < 0:
                    x_opt[i] = 5.0
                else:
                    x_opt[i] = 0.0

            # Evaluate the computed optimum
            f_opt_val = func(x_opt)
            calls_used += 1
            update_best(x_opt, f_opt_val)

            if time.time() > deadline:
                return best_x, best_f

            # Phase 2: Verify with a second point to check linearity
            # Use a fixed seed for reproducibility of the check point
            rng = np.random.default_rng(42)
            x_check = rng.uniform(-4.0, 4.0, size=dim)
            f_check = func(x_check)
            calls_used += 1
            update_best(x_check, f_check)

            f_predicted = w @ x_check + b
            model_error = abs(f_check - f_predicted)

            if time.time() > deadline:
                return best_x, best_f

            # If model error is small, we are confident in the linear model
            # and the vertex x_opt is the global minimum.
            # In this case, further exploration is unlikely to yield improvement.
            # We can spend the remaining budget on a very low-cost verification
            # or simply return.
            
            # However, to be safe against numerical issues or if the "linear" assumption
            # is slightly violated (e.g. by the hidden shift/rotation interacting
            # with clipping in a non-linear way, though the contract says it's linear),
            # we perform a lightweight local search if we have budget.
            
            if model_error < 1e-5:
                # Confident case: Use remaining budget for a tiny-step (1+1)-ES
                # to ensure we haven't missed a better point due to precision.
                x_cur = x_opt.copy()
                f_cur = f_opt_val
                sigma = 1e-3
                
                remaining = budget - calls_used
                # Limit the number of extra steps to avoid overhead, 
                # though 500 is small.
                for _ in range(remaining):
                    if time.time() > deadline:
                        break

                    y = x_cur + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls_used += 1
                    update_best(y, fy)

                    if fy < f_cur:
                        x_cur = y
                        f_cur = fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.9

                    if sigma < 1e-8:
                        break

            else:
                # Uncertain case: Use a more aggressive (1+1)-ES
                x_cur = x_opt.copy()
                f_cur = f_opt_val
                sigma = 0.5

                remaining = budget - calls_used
                for _ in range(remaining):
                    if time.time() > deadline:
                        break

                    y = x_cur + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls_used += 1
                    update_best(y, fy)

                    if fy < f_cur:
                        x_cur = y
                        f_cur = fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.9

                    if sigma < 1e-8:
                        break

            return best_x, best_f
    # EVOLVE-BLOCK-END
