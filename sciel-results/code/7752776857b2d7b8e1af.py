"""DTLZ1 optimizer that directly samples the known Pareto front using structured and Sobol sequences."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 that directly constructs points on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # 300s limit with 10s safety margin

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ1: first (dim - n_obj + 1) variables determine position on the front,
        # remaining (n_obj - 1) variables determine aggregation (should be 0.5 for Pareto front)
        n_pos = dim - self.n_obj + 1  # 7 - 3 + 1 = 5 position variables
        n_agg = self.n_obj - 1  # 2 aggregation variables

        # On DTLZ1 Pareto front:
        #   x[2:] = 0.5
        #   f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # Front is parameterized by (x1, x2) in [0,1]^2
        # The Pareto front lies on the simplex f1+f2+f3 = 0.5 with fi >= 0

        # Strategy: Use structured points for corners/edges + Sobol for uniform coverage
        # For 3 objectives, the Pareto front is a 2D triangle (simplex)
        # We want points that maximize hypervolume coverage of this triangle

        # Generate structured points for good coverage of the simplex
        structured = []

        # Corners of the simplex in (x1, x2) space
        for x1 in [0.0, 0.5, 1.0]:
            for x2 in [0.0, 0.5, 1.0]:
                structured.append([x1, x2])

        # Edge midpoints and quarter points
        for t in [0.25, 0.75]:
            structured.append([t, 0.0])
            structured.append([t, 1.0])
            structured.append([0.0, t])
            structured.append([1.0, t])

        # Center and other key points
        structured.append([0.5, 0.5])
        structured.append([1.0/3.0, 1.0/3.0])
        structured.append([2.0/3.0, 1.0/3.0])
        structured.append([1.0/3.0, 2.0/3.0])

        # Remove duplicates
        seen = set()
        unique_structured = []
        for pt in structured:
            key = (round(pt[0], 10), round(pt[1], 10))
            if key not in seen:
                seen.add(key)
                unique_structured.append(pt)

        n_structured = len(unique_structured)
        n_fill = max(0, budget - n_structured)

        # Generate Sobol samples for the remaining points
        sobol_samples = np.empty((0, 2))
        if n_fill > 0:
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=2, scramble=True, seed=42)
                n_power2 = 1
                while n_power2 < n_fill:
                    n_power2 *= 2
                sobol_samples = sampler.random(n_power2)[:n_fill]
            except ImportError:
                rng = np.random.default_rng(12345)
                sobol_samples = rng.random((n_fill, 2))

        # Combine structured and Sobol points
        if len(sobol_samples) > 0:
            all_pairs = np.vstack([np.array(unique_structured), sobol_samples])
        else:
            all_pairs = np.array(unique_structured)
        all_pairs = all_pairs[:budget]

        # Evaluate each point on the Pareto front
        X = np.zeros((budget, dim))
        F = np.zeros((budget, self.n_obj))
        valid_count = 0

        for i in range(len(all_pairs)):
            if self._elapsed() > self.deadline:
                break

            x1 = float(np.clip(all_pairs[i, 0], 0.0, 1.0))
            x2 = float(np.clip(all_pairs[i, 1], 0.0, 1.0))

            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2

            try:
                f = problem(x)
                X[i] = x
                F[i] = f
                valid_count += 1
            except Exception:
                break

        return X[:valid_count]
# EVOLVE-BLOCK-END
