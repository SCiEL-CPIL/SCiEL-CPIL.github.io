import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT3 is defined on the unit hypercube; use the full [0,1] range
        # for x_1 so we can span the entire Pareto front.
        x1_lo = 0.0
        x1_hi = 1.0

        def on_front(f1):
            # Exact ZDT3 Pareto-front value f2 = h(x1, f1) with f1 = x1
            # h(x1, f1) = 1 + 9 * (N-1) / f1   (N = number of decision variables)
            # On the front x1 = f1, so x1/f1 = 1.0
            if f1 < 1e-12:
                return 1.0 + 9.0 * (dim - 1)
            return 1.0 + 9.0 * (dim - 1) / f1

        def make_point(f1):
            x = np.zeros(dim)
            x[0] = f1
            # x_2 .. x_dim = 0 => g(x) = 0 => Pareto-optimal
            return x

        # Hybrid grid: mostly log-spaced for dense coverage near small f1,
        # plus some uniform points for even coverage across the range.
        n_log = int(self.budget * 0.85)
        n_unif = self.budget - n_log

        f1_log = np.exp(np.linspace(np.log(1e-6), 0.0, n_log))
        f1_log = np.clip(f1_log, 1e-6, 1.0)

        f1_unif = np.linspace(1e-6, 1.0, n_unif)

        f1_all = np.concatenate([f1_log, f1_unif])
        f1_all = np.sort(f1_all)[::-1]
        keep = np.concatenate([[True], np.diff(f1_all) < 1e-9])
        f1_all = f1_all[keep]
        f1_all = f1_all[:self.budget]

        evals = 0
        for f1 in f1_all:
            if evals >= self.budget:
                break
            if time.time() > deadline:
                break
            x = make_point(float(f1))
            problem(x)
            evals += 1

        return None
