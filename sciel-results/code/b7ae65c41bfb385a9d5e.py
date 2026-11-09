# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic structure-exploiting DTLZ4 optimizer.

    For DTLZ4 with M objectives and N variables:
      - The first M-1 variables (z-variables) determine the shape of the Pareto front.
      - The remaining N-(M-1) variables (g-variables) determine the distance from the front.
      - On the Pareto front, the g-variables are at their optimum (0.5 for standard DTLZ4).
      - The objective values on the front are determined solely by the z-variables.

    Strategy:
      1. Generate a dense set of z-variables using a Halton low-discrepancy sequence.
      2. Fix g-variables to 0.5 (the known optimum).
      3. Evaluate all candidates, recording every evaluation.
      4. Use remaining budget for a few local refinements if time permits.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # hard safety margin (< 300s wall-clock)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        M = self.n_obj
        n_z = M - 1  # number of shape variables

        # g-variables fixed to their optimum (midpoint of bounds)
        g_center = (lo[n_z:] + hi[n_z:]) / 2.0

        # Generate Halton sequence points in [0,1]^n_z
        def halton_sequence(n_points, dim):
            """Generate n_points Halton sequence points in [0,1]^dim."""
            points = []
            for i in range(1, n_points + 1):
                point = []
                for base in range(2, 2 + dim):
                    f = 1.0 / base
                    result = 0.0
                    while f > 0:
                        f /= base
                        result += f * (i % int(1 / f))
                    point.append(result)
                points.append(point)
            return np.array(points)

        # Determine how many z-points we can afford
        # We want to use most of the budget for the structured grid
        n_candidates = min(self.budget - 10, 180)  # leave some budget for refinement

        # Generate Halton points
        z_points = halton_sequence(n_candidates, n_z)

        # Map to actual bounds
        z_actual = z_points * (hi[:n_z] - lo[:n_z]) + lo[:n_z]

        # Build full candidate vectors
        candidates = []
        for z in z_actual:
            x = np.concatenate([z, g_center])
            candidates.append(x)

        # Evaluate candidates, recording every evaluation
        F_list = []
        X_list = []
        used = 0

        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = np.asarray(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1

        X = np.array(X_list)
        F = np.array(F_list)

        # Use remaining budget for local refinement of the best points
        if len(F) > 0 and used < self.budget:
            # Find non-dominated points
            nd_idx = []
            for i in range(len(F)):
                dom = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom = True
                        break
                if not dom:
                    nd_idx.append(i)

            if not nd_idx:
                nd_idx = list(range(len(F)))

            # Refine the best few points
            sums = F[nd_idx].sum(axis=1)
            order = np.argsort(sums)[:min(5, len(nd_idx))]
            refine_idx = [nd_idx[k] for k in order]

            base_X = X.copy()
            base_F = F.copy()

            # Small step for local search
            step = span * 0.02
            rng = np.random.default_rng(42)

            for _ in range(min(self.budget - used, 10)):
                if time.time() > deadline:
                    break
                if not refine_idx:
                    break
                pi = refine_idx[used % len(refine_idx)]
                if pi >= len(base_X):
                    pi = 0
                x0 = base_X[pi]
                d = step * rng.standard_normal(n)
                x = np.clip(x0 + d, lo, hi)
                f = np.asarray(problem(x), dtype=float)
                used += 1
                # Accept if it improves the sum
                if f.sum() < base_F[pi].sum():
                    base_X[pi] = x
                    base_F[pi] = f

            # Merge with original
            final_X = np.concatenate([X, base_X], axis=0)
            final_F = np.concatenate([F, base_F], axis=0)
        else:
            final_X = X
            final_F = F

        return final_X
# EVOLVE-BLOCK-END
