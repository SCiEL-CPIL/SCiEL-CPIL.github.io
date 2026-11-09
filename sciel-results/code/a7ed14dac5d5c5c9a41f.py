"""Multi-objective optimizer for RE34: Sobol initialization + HV-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol initialization and hypervolume-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # Safety margin before 330s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # ---------- Hypervolume helpers ----------
        def hv2(sub, ref2):
            """2D hypervolume (minimization)."""
            if len(sub) == 0:
                return 0.0
            mask = np.all(sub < ref2, axis=1)
            sub = sub[mask]
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                w0 = x - prev_x
                if w0 <= 0:
                    continue
                ybest = np.min(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                if ybest < ref2[1]:
                    total += w0 * (ref2[1] - ybest)
                prev_x = x
            return total

        def hv3(P, ref3):
            """3D hypervolume (minimization)."""
            if len(P) == 0:
                return 0.0
            mask = np.all(P < ref3, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                w0 = x - prev_x
                if w0 <= 0:
                    continue
                sub = P[i:, 1:]
                total += w0 * hv2(sub, ref3[1:])
                prev_x = x
            return total

        def pareto_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            keep = []
            for i in range(n):
                fi = F[i]
                # A point j dominates i if F[j] <= F[i] in all objectives and < in at least one
                better = np.all(F <= fi, axis=1) & np.any(F < fi, axis=1)
                better[i] = False
                if not np.any(better):
                    keep.append(i)
            return keep

        # ---------- Main Loop ----------
        X = []
        F = []

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # 1. Sobol sequence initialization: 128 points for maximal space-filling
        n_sobol = min(128, budget)
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Sobol requires power-of-2 for best quality; 128 = 2^7
        sobol_points = sobol.random(n_sobol)
        # Map from [0,1]^dim to [lo, hi]
        for i in range(n_sobol):
            x = lo + (hi - lo) * sobol_points[i]
            x = np.clip(x, lo, hi)
            add(x, problem(x))

        used = len(F)

        # 2. Local search on top HV-contributing points
        # We'll iteratively select the point with highest marginal HV contribution
        # and apply small axis-aligned perturbations around it.

        # Adaptive step size
        base_step = 0.15
        min_step = 0.01
        max_step = 0.3

        while used < budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F)

            # Get Pareto front
            pf_idx = pareto_indices(F_arr)
            if not pf_idx:
                pf_idx = list(range(len(F_arr)))

            # Compute marginal HV contribution for each Pareto point
            # Approximation: for each point on the front, estimate how much HV it contributes
            # by checking the region it "covers" relative to neighbors.
            # Simpler heuristic: rank by sum of objectives (lower = better central point)
            pf_F = F_arr[pf_idx]
            # Use a combination: lower objective sum + some randomness
            sums = np.sum(pf_F, axis=1)
            # Add small noise to break ties
            noise = rng.random(len(pf_idx)) * 0.01 * np.max(sums) if len(pf_idx) > 0 else 0
            scores = sums + noise
            # Select top 10 by score (lowest sum)
            n_select = min(10, len(pf_idx))
            top_local = np.argsort(scores)[:n_select]
            # Pick one randomly from top
            pidx_local = top_local[rng.integers(n_select)]
            pidx = pf_idx[pidx_local]

            # Also occasionally pick from all points for exploration
            if rng.random() < 0.15:
                pidx = rng.integers(len(F_arr))

            # Adaptive step: shrink over time, expand if stuck
            progress = used / budget
            step = base_step * (1.0 - 0.5 * progress)
            step = max(min_step, min(step, max_step))

            # Axis-aligned perturbation: pick 1-2 random dimensions to perturb
            n_perturb = rng.integers(1, 3)
            dims_to_perturb = rng.choice(dim, size=n_perturb, replace=False)

            child = X[pidx].copy()
            for d in dims_to_perturb:
                # Small Gaussian perturbation along one axis
                delta = step * (hi[d] - lo[d]) * rng.standard_normal()
                child[d] += delta
            child = np.clip(child, lo, hi)

            fc = problem(child)
            add(child, fc)
            used += 1

        return X
# EVOLVE-BLOCK-END
