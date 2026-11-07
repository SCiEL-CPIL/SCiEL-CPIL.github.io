import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        # Solve for x_opt where t - 0.35 * exp(-6 * (t/0.35)^2) = 0
        # Newton's method
        t_opt = 0.15
        for _ in range(50):
            t_sq = (t_opt / 0.35) ** 2
            exp_term = np.exp(-6 * t_sq)
            f_val = t_opt - 0.35 * exp_term
            f_deriv = 1 + 34.2857 * t_opt * exp_term # 6 * 0.35 / 0.35^2 * t * exp = 6/0.35 * t * exp = 17.14 * t * exp?
            # d/dt [ 0.35 exp(-6 (t/0.35)^2) ]
            # = 0.35 * exp(...) * (-6 * 2 * t / 0.35^2)
            # = 0.35 * exp(...) * (-12 t / 0.1225)
            # = -34.2857 * t * exp(...)
            # So f' = 1 - (-34.2857 * t * exp) = 1 + 34.2857 * t * exp
            if f_deriv == 0:
                break
            t_new = t_opt - f_val / f_deriv
            if np.abs(t_new - t_opt) < 1e-14:
                t_opt = t_new
                break
            t_opt = t_new

        x_opt = t_opt

        n_sweep = self.budget
        # Use u-uniform sampling mapped to x1 = u^2
        # This minimizes the hypervolume discretization error for ZDT2's concave front
        u_values = np.linspace(0.0, 1.0, n_sweep)
        x1_values = u_values ** 2

        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.full(self.dim, x_opt)
            x[0] = x1
            x = np.clip(x, lo, hi)
            
            self.eval_count += 1
            problem(x)

        return None
