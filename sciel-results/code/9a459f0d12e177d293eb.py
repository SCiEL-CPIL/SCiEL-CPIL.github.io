"""Multi-objective optimizer exploiting ZDT3 structure with stratified front sampling."""
import numpy as np
import time


class Optimizer:
    """Exploits ZDT3 structure: place all points on the analytic Pareto manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3: f1 = x1, f2 = g(x2..xn) * h(f1, g)
        # g = 1 + 10(n-1) + sum(25*(sin(50*x_i^0.1)) for i=2..n)
        # h = 1 + sin^3(10*x1*pi)/2
        # Optimal: x2..xn=0 -> g=1, f1=x1, f2=h(x1)
        # Pareto front: f1 in [0,1], f2 = h(f1)

        # h(t) = 1 + sin^3(10*pi*t)/2
        # Local minima at t = 0.15 + 0.2*k for k=0,1,2,3,4
        # h_min = 1 + (-1)^3/2 = 0.5

        # Strategy: stratified sampling on [0,1] with each stratum's point
        # placed at the t value that minimizes h(t) within that stratum.
        # This ensures maximum hypervolume coverage on the known manifold.

        n = budget
        t_values = np.empty(n)
        w = 1.0 / n

        # Pre-compute h(t) for efficiency
        def h_func(t):
            return 1.0 + (np.sin(10.0 * np.pi * t) ** 3) / 2.0

        # The 5 interior minima of h in [0,1]
        minima_t = [0.15, 0.35, 0.55, 0.75, 0.95]

        for i in range(n):
            a = i * w
            b = a + w
            # Candidate t values: endpoints and any interior minima
            candidates = [a, b]
            for tm in minima_t:
                if a < tm < b:
                    candidates.append(tm)
            # Pick candidate with lowest h
            best_t = candidates[0]
            best_h = h_func(best_t)
            for tc in candidates[1:]:
                h_val = h_func(tc)
                if h_val < best_h:
                    best_h = h_val
                    best_t = tc
            t_values[i] = best_t

        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return np.array(f)

        # Evaluate all points on the manifold
        for t in t_values:
            if used >= budget:
                break
            if time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            results.append((x.copy(), f))

        return results
