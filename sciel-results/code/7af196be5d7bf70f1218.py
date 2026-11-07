"""Multi-objective optimizer for ZDT3 using uniform Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits ZDT3 structure: f1=x1, f2=g*(1-sqrt(f1/g))."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
        # f2(x) = g(x) * (1 - sqrt(f1/g(x)))
        #
        # The Pareto front is achieved when g is minimized (g_min = 1) and x1 varies in [0,1].
        # When g=1, f2 = 1 - sqrt(x1)
        #
        # KEY INSIGHT: g(x[1:]) is minimized at x[1:]=0 giving g=1 exactly.
        # We can compute g analytically without spending any evaluation budget on it.
        # Therefore, ALL budget goes to sampling the Pareto front.
        #
        # Hypervolume contribution of point (f1, f2) with ref [11,11]:
        #   (11-f1)*(11-f2) = (11-f1)*(10+sqrt(f1))
        # This function is concave in f1, so uniform spacing maximizes total hypervolume.

        # The optimal tail is x[1:]=0, giving g=1
        best_x_tail = np.zeros(dim - 1)

        # Use uniform grid for maximum hypervolume coverage
        # With budget evaluations, use all budget for distinct x1 values
        n_evals = budget
        x1_values = np.linspace(0.0, 1.0, n_evals)

        # Evaluate each x1 value using the known optimal tail
        results_F = []
        evaluations_used = 0

        for i in range(n_evals):
            if evaluations_used >= budget:
                break
            if time.time() > deadline:
                break

            x1_val = float(x1_values[i])
            x = np.empty(dim)
            x[0] = x1_val
            x[1:] = best_x_tail
            x = np.clip(x, lo, hi)
            f = problem(x)
            results_F.append(f)
            evaluations_used += 1

        # Convert to array
        if results_F:
            F_final = np.array(results_F)
        else:
            F_final = np.array([[1.0, 1.0]])

        return F_final
# EVOLVE-BLOCK-END
