"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS Init + Weighted-Sum L-BFGS-B Multi-start Polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_list = list(zip(lo, hi))

        # Phase 1: Latin Hypercube Sampling for initial spread
        n_lhs = 50
        X_lhs = np.empty((n_lhs, self.dim))
        for d in range(self.dim):
            X_lhs[:, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(n_lhs) + rng.random()) / n_lhs
        rng.shuffle(X_lhs)

        archive = []
        used = 0

        def add_point(x, f):
            nonlocal used
            if used >= self.budget:
                return
            if x is None:
                return
            x = np.clip(x, lo, hi)
            if f is None:
                f = problem(x)
            archive.append((x, f))
            used += 1

        # Evaluate LHS points
        for x in X_lhs:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            add_point(x, f)

        # Phase 2: Multi-start L-BFGS-B on weighted sums
        # Use a set of weight vectors to explore different parts of the Pareto front
        n_weights = 10
        # Generate weight vectors on the simplex
        weights = []
        for i in range(n_weights):
            w = np.random.dirichlet(np.ones(self.n_obj), size=1)[0]
            weights.append(w)

        # Also add some extreme weights (unit vectors)
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Shuffle weights to randomize order
        rng.shuffle(weights)

        for w in weights:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Objective: minimize weighted sum
            def weighted_obj(x, w=w):
                f = problem(x)
                return float(np.dot(w, f))

            # Use multiple starting points from archive
            starts = []
            # Add a few random starts
            for _ in range(3):
                x0 = lo + (hi - lo) * rng.random(self.dim)
                starts.append(x0)
            # Add some archive points
            if len(archive) > 0:
                idxs = rng.choice(len(archive), size=min(3, len(archive)), replace=False)
                for idx in idxs:
                    starts.append(archive[idx][0].copy())

            best_result = None
            for x0 in starts:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                try:
                    res = minimize(weighted_obj, x0, method='L-BFGS-B',
                                   bounds=bounds_list,
                                   options={'maxiter': 200, 'maxfun': 50})
                    if res.success and res.fun < np.inf:
                        x_opt = np.clip(res.x, lo, hi)
                        f_opt = problem(x_opt)
                        if best_result is None or res.fun < best_result[1]:
                            best_result = (x_opt, f_opt, res.fun)
                except Exception:
                    pass

            if best_result is not None:
                add_point(best_result[0], best_result[1])

        return archive
# EVOLVE-BLOCK-END
