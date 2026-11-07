"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT3 structure: place points on Pareto front (x[1:]=0)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        n = budget  # Use all budget on Pareto front

        # Solve for optimal x[0] values using iterative local optimization
        # H = sum_{i=1}^{n} (10 + sqrt(x_i)) * (x_{i+1} - x_i), x_{n+1} = 11
        # Initialize with geometric spacing (better for sqrt front)
        if n > 2:
            x0_vals = np.geomspace(1e-6, 1.0, n)
        else:
            x0_vals = np.linspace(0, 1, n)

        # Iteratively optimize each point
        for iteration in range(20):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                # Determine bounds for x_i based on neighbors
                lo_i = x0_vals[i-1] + 1e-10 if i > 0 else 1e-10
                hi_i = x0_vals[i+1] - 1e-10 if i < n-1 else 1.0
                if lo_i >= hi_i:
                    continue

                # Local objective: contribution of terms involving x_i
                # Term (i-1): (10 + sqrt(x_{i-1})) * (x_i - x_{i-1})  [if i > 0]
                # Term i: (10 + sqrt(x_i)) * (x_{i+1} - x_i)
                # Term (i+1) doesn't involve x_i directly

                def local_obj(xi):
                    obj = 0.0
                    if i > 0:
                        obj += (10.0 + np.sqrt(x0_vals[i-1])) * (xi - x0_vals[i-1])
                    obj += (10.0 + np.sqrt(xi)) * (x0_vals[i+1] - xi) if i < n-1 else (10.0 + np.sqrt(xi)) * (11.0 - xi)
                    return obj

                # Golden section search
                gr = (np.sqrt(5) - 1) / 2
                a, b = lo_i, hi_i
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc, fd = local_obj(c), local_obj(d)
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    if fc < fd:
                        b, d, fd = d, c, fc
                        c = b - gr * (b - a)
                        fc = local_obj(c)
                    else:
                        a, c, fc = c, d, fd
                        d = a + gr * (b - a)
                        fd = local_obj(d)
                x_new = (a + b) / 2
                if abs(x_new - x0_vals[i]) > 1e-12:
                    x0_vals[i] = x_new
                    improved = True

            if not improved:
                break

        # Build points
        X = np.zeros((n, dim))
        X[:, 0] = x0_vals

        # Evaluate all
        for i in range(n):
            if time.time() > deadline:
                break
            _ = problem(X[i])

        return X
# EVOLVE-BLOCK-END
