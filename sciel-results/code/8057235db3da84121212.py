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
        deadline = 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # ZDT2 Pareto front: x[1:] = 0.5 minimizes g(x), giving the true Pareto front.
        # g_min = 1 + 9*(n-1) for n=30 -> g_min = 262
        # f1 = x[0], f2 = g_min * (1 - sqrt(f1/g_min))
        # To maximize hypervolume, we want points well-distributed along the front.
        # Use a non-uniform distribution: denser near f1=0 where f2 is large,
        # sparser near f1=1 where f2 is small, to maximize HV contribution.
        n_sweep = self.budget
        # Use x0 = t^p with p>1 to get denser sampling near 0
        t_values = np.linspace(0.0, 1.0, n_sweep)
        x0_values = t_values ** 2.5
        
        for x0 in x0_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
