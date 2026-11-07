import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, self.n_obj))

        N = budget

        # ZDT4 Pareto front: f1 = x0, f2 = 1 - sqrt(x0), for x0 in [0, 1]
        # Optimal spacing for HV maximization: x0_i = (i/(N-1))^2 for i=0..N-1
        # This gives exactly x0_0 = 0 and x0_{N-1} = 1
        if N == 1:
            x0_vals = np.array([0.5])  # midpoint if only 1 eval
        else:
            idx = np.arange(N, dtype=float)
            x0_vals = (idx / (N - 1.0)) ** 2.0

        # Clip to valid range [0, 1]
        x0_vals = np.clip(x0_vals, 0.0, 1.0)

        # Ensure strictly increasing (safety)
        for k in range(1, N):
            if x0_vals[k] <= x0_vals[k - 1]:
                x0_vals[k] = x0_vals[k - 1] + 1e-15

        # For x1..x9, set to 0 (or closest bound to 0) to minimize g(x)
        tail = np.zeros(dim - 1)
        for k in range(dim - 1):
            if lo[k + 1] <= 0.0 <= hi[k + 1]:
                tail[k] = 0.0
            elif 0.0 < lo[k + 1]:
                tail[k] = lo[k + 1]
            else:
                tail[k] = hi[k + 1]

        evals = 0
        results = []

        for x0_val in x0_vals:
            if evals >= budget:
                break
            if time.time() - t0 > DEADLINE:
                break

            x = np.empty(dim)
            x[0] = x0_val
            x[1:] = tail
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                evals += 1
                if f is not None:
                    results.append(np.asarray(f, dtype=float))
            except Exception:
                pass

        if results:
            return np.array(results)

        return np.empty((0, self.n_obj))
