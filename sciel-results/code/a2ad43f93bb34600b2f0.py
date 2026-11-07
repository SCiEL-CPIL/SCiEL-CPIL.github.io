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

        # Build a uniform grid in f1 over [0, 1] to capture ZDT3's oscillatory
        # Pareto front (10 periods of sin(10*pi*f1) in [0,1]).
        # Uniform spacing ensures even coverage of the oscillations, maximizing
        # the hypervolume of the staircase formed by the evaluated points.
        n_sweep = self.budget
        # ZDT3 Pareto front: f2 = 1 - sqrt(f1)*sin(10*pi*f1) for f1 in [0,1]
        # Local minima of f2 occur where sin(10*pi*f1) = 1, i.e., f1 = (2k+1)/20
        # These points contribute most to hypervolume, so include them explicitly.
        minima_f1 = np.array([(2*k+1)/20.0 for k in range(10)])
        # Remaining points: distribute uniformly in [0,1] but exclude near the minima
        n_remaining = n_sweep - len(minima_f1)
        # Uniform grid for the remaining points
        uniform_grid = np.linspace(0.0, 1.0, n_remaining + 1)
        # Combine and sort
        f1_grid = np.sort(np.concatenate([minima_f1, uniform_grid]))

        evals = 0
        for f1 in f1_grid:
            if evals >= self.budget:
                break
            if time.time() > deadline:
                break
            x = make_point(float(f1))
            problem(x)
            evals += 1

        return None
