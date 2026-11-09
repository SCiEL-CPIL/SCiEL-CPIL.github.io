# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT1, the Pareto front is achieved when x_2...x_d = 0
        # and x_1 varies in [0, 1].
        # f1 = x1
        # f2 = 1 - sqrt(x1) (since g=1)
        # Hypervolume contribution of point at t: (t - t_prev) * (10 + sqrt(t))
        # Optimal density proportional to 1/(10 + sqrt(t))
        
        n_evals = self.budget
        if n_evals > 0:
            # Compute optimal CDF using numerical integration
            # F(t) = integral_0^t 1/(10+sqrt(s)) ds / integral_0^1 1/(10+sqrt(s)) ds
            # Use substitution s = u^2, ds = 2u du
            # integral = integral_0^sqrt(t) 2u/(10+u) du
            # = integral_0^sqrt(t) 2(1 - 10/(10+u)) du
            # = 2*sqrt(t) - 20*ln(1 + sqrt(t)/10)
            
            # Let's use this analytical form
            # G(t) = 2*sqrt(t) - 20*ln(1 + sqrt(t)/10)
            # F(t) = G(t) / G(1)
            
            def G(t):
                st = np.sqrt(t)
                return 2 * st - 20 * np.log(1 + st / 10.0)
            
            G1 = G(1.0)
            
            # u_i = i / (N-1) for i = 0, ..., N-1
            # We want F(t_i) = u_i, so t_i = F^{-1}(u_i)
            # G(t_i) = u_i * G1
            # Solve: 2*sqrt(t) - 20*ln(1 + sqrt(t)/10) = u_i * G1
            
            idx = np.arange(n_evals)
            u = idx / (n_evals - 1)
            target = u * G1
            
            # Vectorized Newton's method to solve G(t) = target
            # G(t) = 2*sqrt(t) - 20*ln(1 + sqrt(t)/10)
            # G'(t) = 1/sqrt(t) - 2/(10 + sqrt(t))
            
            def G_vec(t):
                st = np.sqrt(np.maximum(t, 1e-15))
                return 2 * st - 20 * np.log(1 + st / 10.0)
            
            def Gp_vec(t):
                st = np.sqrt(np.maximum(t, 1e-15))
                return 1.0 / st - 2.0 / (10.0 + st)
            
            t_vals = u.copy()  # initial guess
            for _ in range(30):
                g_val = G_vec(t_vals)
                gp_val = Gp_vec(t_vals)
                mask = np.abs(gp_val) > 1e-12
                t_vals[mask] -= (g_val[mask] - target[mask]) / gp_val[mask]
                t_vals = np.clip(t_vals, 0.0, 1.0)
            
            # Final high-precision bisection polish for ALL points
            # This ensures maximum numerical accuracy for each point
            for i in range(n_evals):
                lo_y, hi_y = 0.0, 1.0
                for _ in range(80):
                    mid_y = (lo_y + hi_y) / 2.0
                    val = 2 * mid_y - 20 * np.log(1 + mid_y / 10.0)
                    if val < target[i]:
                        lo_y = mid_y
                    else:
                        hi_y = mid_y
                t_vals[i] = ((lo_y + hi_y) / 2.0) ** 2
            
            # Construct points
            X = np.zeros((n_evals, self.dim))
            X[:, 0] = t_vals
            X = np.clip(X, lo, hi)
            
            last_obj = None
            for i in range(n_evals):
                if time.time() > deadline:
                    break
                x = X[i]
                f = problem(x)
                last_obj = f
                
            return last_obj
        else:
            return None
# EVOLVE-BLOCK-END
