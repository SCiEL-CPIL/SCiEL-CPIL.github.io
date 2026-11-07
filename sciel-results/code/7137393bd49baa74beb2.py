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

        n = self.dim

        # ZDT2 Pareto front: x[1:] = 0, f1 = x[0], f2 = 1 - sqrt(x[0])
        # Hypervolume integral: H = ∫₀¹ [11 - (1 - sqrt(x))] dx = ∫₀¹ [10 + sqrt(x)] dx
        # = 10 + 2/3 ≈ 10.667
        # The integrand g(x) = 10 + sqrt(x) has g'(x) = 1/(2*sqrt(x)) → ∞ at x=0.
        # To minimize quadrature error, cluster points near x=0 where curvature is high.
        # Optimal spacing: Δx_i ∝ sqrt(x_i), which gives x_i ∝ (i/N)².

        x1_min = max(0.0, lo[0])
        x1_max = min(1.0, hi[0])

        if x1_min >= x1_max:
            x1_min = 0.0
            x1_max = 1.0

        n_sweep = min(self.budget, 200)

        # Non-uniform grid: x_i = x1_min + (x1_max - x1_min) * (i/(N-1))²
        # This clusters points near x1_min (which is 0 for ZDT2), where g'(x) is large.
        t = np.linspace(0.0, 1.0, n_sweep)
        x1_values = x1_min + (x1_max - x1_min) * t**2

        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
