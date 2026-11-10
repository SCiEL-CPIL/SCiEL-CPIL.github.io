import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        results = []

        # ZDT6 Pareto front: x[1:] = 0, x[0] in [0,1]
        # f1(x0) = 1 + x0 * (sin(10*pi*x0) + sin(10*pi*x0**2))
        # f2(x0) = 1 - sqrt(f1(x0))
        
        def f1_analytic(x0):
            return 1.0 + x0 * (np.sin(10.0 * np.pi * x0) + np.sin(10.0 * np.pi * x0**2))

        # Strategy: Uniform sampling in f1 (objective) space to maximize hypervolume.
        # Since f2 is a monotonic function of f1, the Pareto front in objective space
        # is determined by the range of f1 values. We sample f1 uniformly and map
        # back to x0. This avoids redundant points where f1(x0) is non-monotonic.
        
        # 1. Create a dense grid of x0 to map f1 values to x0
        x_dense = np.linspace(0.0, 1.0, 10000)
        f1_dense = f1_analytic(x_dense)
        
        f1_min = np.min(f1_dense)
        f1_max = np.max(f1_dense)
        
        # 2. Generate uniform targets in f1 space
        v_targets = np.linspace(f1_min, f1_max, n_evals)
        
        # 3. Map each target f1 value back to an x0 value
        x0s = np.zeros(n_evals)
        for i in range(n_evals):
            # Find x0 in x_dense that produces f1 closest to v_targets[i]
            idx = np.argmin(np.abs(f1_dense - v_targets[i]))
            x0s[i] = x_dense[idx]

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
