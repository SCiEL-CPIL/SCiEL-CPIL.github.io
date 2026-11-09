"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective optimizer with stratified initialization and adaptive sigma."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Stratified initialization using Latin Hypercube Sampling
        n_init = min(50, budget // 4)
        rng = np.random.default_rng(42)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + rng.random(n_init)) / n_init
        X = lo + (hi - lo) * X

        F = np.array([problem(x) for x in X], dtype=float)
        used = n_init

        # Reference point for hypervolume (used internally for selection)
        ref = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for points dominated by ref."""
            # Filter points that are better than ref in all objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Remove dominated points (keep Pareto front)
            pareto = [pts[0]]
            for i in range(1, len(pts)):
                if pts[i, 1] < pareto[-1][1]:
                    pareto.append(pts[i])
            pareto = np.array(pareto)
            if len(pareto) == 0:
                return 0.0
            # Compute hypervolume
            hv = 0.0
            for i in range(len(pareto)):
                x1 = pareto[i, 0]
                x2 = pareto[i, 1]
                # Width in x1 direction
                if i < len(pareto) - 1:
                    w = pareto[i + 1, 0] - x1
                else:
                    w = ref[0] - x1
                # Height in x2 direction
                h = ref[1] - x2
                hv += w * h
            return hv

        def hv_contribution(point, points, ref):
            """Compute marginal hypervolume contribution of a point."""
            # HV without the point
            others = [p for p in points if not np.allclose(p, point)]
            if len(others) == 0:
                return hypervolume_2d(np.array([point]), ref)
            hv_with = hypervolume_2d(np.array([point] + others), ref)
            hv_without = hypervolume_2d(np.array(others), ref)
            return hv_with - hv_without

        # Adaptive sigma for mutation
        sigma = 0.3  # initial step size relative to bounds

        while used < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Select parent using HypE-style marginal hypervolume contribution
            # Use a subset of points for efficiency
            if len(F) <= 20:
                candidates = list(range(len(F)))
            else:
                candidates = list(range(len(F)))

            # Compute contributions for a random subset for speed
            if len(candidates) > 30:
                sample_idx = rng.choice(len(candidates), size=30, replace=False)
                sample_candidates = [candidates[i] for i in sample_idx]
            else:
                sample_candidates = candidates

            # Compute marginal HV contributions
            contributions = np.zeros(len(sample_candidates))
            for i, idx in enumerate(sample_candidates):
                pt = F[idx]
                others = [F[j] for j in range(len(F)) if j != idx]
                if len(others) == 0:
                    contributions[i] = hypervolume_2d(pt.reshape(1, -1), ref)
                else:
                    others_arr = np.array(others)
                    all_pts = np.vstack([pt.reshape(1, -1), others_arr])
                    contributions[i] = hypervolume_2d(all_pts, ref) - hypervolume_2d(others_arr, ref)

            # Select parent with highest contribution
            best_local = int(np.argmax(contributions))
            parent_idx = sample_candidates[best_local]

            # Mutate
            sigma_decay = max(0.01, 0.3 * (1.0 - used / budget))
            child = X[parent_idx] + sigma_decay * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point in archive
            # Use sum of normalized objectives as simple fitness
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
