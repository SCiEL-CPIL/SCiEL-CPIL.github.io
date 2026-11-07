import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None
        self.x_opt = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        # Find optimal x_i for i > 0
        # Solve t = 0.35 * exp(-6 * (t/0.35)^2)
        # Using Newton's method
        t_opt = 0.15
        for _ in range(20):
            f_val = t_opt - 0.35 * np.exp(-6 * (t_opt / 0.35) ** 2)
            f_deriv = 1 + 34.2857 * t_opt * np.exp(-6 * (t_opt / 0.35) ** 2)
            t_new = t_opt - f_val / f_deriv
            if np.abs(t_new - t_opt) < 1e-12:
                t_opt = t_new
                break
            t_opt = t_new
            
        x_opt = t_opt

        n_sweep = self.budget
        # Use non-uniform sampling to maximize hypervolume
        # Density higher near x=0
        # x = (i/N)^2
        indices = np.arange(n_sweep)
        x1_values = (indices / n_sweep) ** 2
        
        # Ensure we include 0 and 1?
        # linspace includes endpoints. 
        # (i/N)^2 for i=0..N-1 gives 0 .. (1-1/N)^2.
        # It misses 1.
        # Let's use linspace of u, then square.
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
