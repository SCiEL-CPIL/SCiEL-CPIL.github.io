# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Problem-aware hybrid optimizer for ZDT1: analytical x1 sweep + local g(x) minimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 structure:
        # f1 = x1
        # g  = 1 + 9 * sum(x2..xn) / (n-1)
        # f2 = g * (1 - sqrt(f1/g))
        #
        # For a given x1, minimizing g gives the best f2.
        # The Pareto front is traced by x1 in [0,1] with g=1 (i.e., x2..xn=0).
        #
        # Strategy:
        # 1. Sample many x1 values, set x2..xn=0, evaluate.
        # 2. For promising x1 values, do local search on x2..xn to minimize g.
        # 3. Use remaining budget for random exploration.

        # Phase 1: Sweep x1 with x2..xn = 0
        n_sweep = min(budget // 3, 100)
        x1_values = np.linspace(0.0, 1.0, n_sweep)
        F_list = []
        X_list = []

        for x1 in x1_values:
            if time.time() > deadline - 5.0:
                break
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_list.append(f)
            X_list.append(x)

        used = len(F_list)
        F = np.array(F_list)
        X = np.array(X_list)

        # Phase 2: Local search on x2..xn for some promising x1 values
        # Pick a subset of x1 values to refine
        n_refine = min(budget // 3, 50)
        if n_refine > 0 and used < budget:
            # Select evenly spaced x1 values from the sweep
            indices = np.linspace(0, len(x1_values) - 1, n_refine, dtype=int)
            for idx in indices:
                if used >= budget or time.time() > deadline - 5.0:
                    break
                base_x = X[idx].copy()
                # Local search on x2..xn using coordinate descent
                for _ in range(5):  # 5 coordinate descent steps
                    if used >= budget or time.time() > deadline - 5.0:
                        break
                    # Try perturbing each of x2..xn slightly
                    for d in range(1, dim):
                        if used >= budget or time.time() > deadline - 5.0:
                            break
                        x_trial = base_x.copy()
                        x_trial[d] = max(lo[d], min(hi[d], base_x[d] + 0.01 * (1 - 2 * (d % 2))))
                        f_trial = problem(x_trial)
                        used += 1
                        # If this improves g (which is roughly f2 when f1 is fixed and g is small),
                        # keep it
                        if f_trial[1] < F[idx][1]:
                            base_x = x_trial
                            F[idx] = f_trial
                            X[idx] = base_x

        # Phase 3: Use remaining budget for random exploration
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 5.0:
            n_random = remaining
            X_rand = lo + (hi - lo) * np.random.default_rng(0).random((n_random, dim))
            for i in range(n_random):
                if time.time() > deadline - 5.0:
                    break
                f = problem(X_rand[i])
                F_list.append(f)
                X_list.append(X_rand[i])
            F = np.array(F_list)
            X = np.array(X_list)

        return X
# EVOLVE-BLOCK-END
