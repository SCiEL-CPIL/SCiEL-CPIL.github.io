"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Decomposition-based multi-objective optimizer using weighted-sum + local Nelder-Mead."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        eval_count = [0]

        def eval_with_count(x):
            if time.time() > deadline:
                raise TimeoutError()
            eval_count[0] += 1
            return np.array(problem(x), dtype=float)

        # Generate weight vectors covering the trade-off space
        # For 2 objectives, weights are (w, 1-w)
        n_weights = 25
        # Use a mix of uniform and edge weights
        weights = []
        # Include extreme weights for the two objectives
        weights.append(np.array([0.0, 1.0]))
        weights.append(np.array([1.0, 0.0]))
        # Uniform weights in between
        for i in range(1, n_weights - 1):
            w = i / (n_weights - 1)
            weights.append(np.array([w, 1.0 - w]))
        weights = np.array(weights)

        # Initial random samples to find good starting points
        rng = np.random.default_rng(42)
        n_init = min(20, budget // 10)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([eval_with_count(x) for x in X_init])

        # Find non-dominated points from initial samples
        def is_non_dominated(F):
            n = len(F)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd[i] = True
            return nd

        nd_mask = is_non_dominated(F_init)
        X_nd = X_init[nd_mask] if np.any(nd_mask) else X_init[:1]
        F_nd = F_init[nd_mask] if np.any(nd_mask) else F_init[:1]

        # For each weight, run local optimization starting from the best-matching initial point
        # Weighted sum objective: minimize sum(w_i * f_i)
        # We'll use Nelder-Mead with bounded constraints handled by clipping

        results = []  # List of (x, f) pairs
        used_budget = eval_count[0]

        # Sort weights to process more "central" ones first (they may be more important for hypervolume)
        # Actually, process all - the decomposition ensures coverage

        for w in weights:
            if time.time() > deadline - 10.0:
                break
            if used_budget >= budget:
                break

            # Find the best starting point: minimize weighted sum among initial non-dominated points
            ws_init = F_nd @ w
            start_idx = np.argmin(ws_init)
            x0 = X_nd[start_idx].copy()

            # Define the scalarized objective
            def obj(x):
                if time.time() > deadline:
                    raise TimeoutError()
                x_clipped = np.clip(x, lo, hi)
                f = np.array(problem(x_clipped), dtype=float)
                eval_count[0] += 1
                used_budget = eval_count[0]
                return np.dot(w, f)

            # Run Nelder-Mead
            try:
                res = minimize(
                    obj,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                x_opt = np.clip(res.x, lo, hi)
                f_opt = np.array(problem(x_opt), dtype=float)
                eval_count[0] += 1
                results.append((x_opt, f_opt))
            except Exception:
                # If optimization fails, keep the starting point
                results.append((x0, F_nd[start_idx]))

            used_budget = eval_count[0]

        # Also add the initial non-dominated points
        for i in range(len(X_nd)):
            results.append((X_nd[i], F_nd[i]))

        # Deduplicate and filter: keep only points strictly better than reference [3000, 0.06]
        ref = np.array([3000.0, 0.06])
        unique_results = []
        seen = set()
        for x, f in results:
            # Check if strictly better than reference in every objective
            if np.all(f < ref):
                key = tuple(np.round(f, 10))
                if key not in seen:
                    seen.add(key)
                    unique_results.append((x, f))

        # If we have budget left, do a final polish: perturb each result and try to improve
        if eval_count[0] < budget:
            for idx in range(len(unique_results)):
                if time.time() > deadline - 5.0:
                    break
                if eval_count[0] >= budget:
                    break
                x_cur, f_cur = unique_results[idx]
                # Try a few small perturbations
                for _ in range(3):
                    if eval_count[0] >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    perturbation = 0.05 * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x_cur + perturbation, lo, hi)
                    f_new = np.array(problem(x_new), dtype=float)
                    eval_count[0] += 1
                    # Check if new point dominates or is non-dominated with current
                    if np.all(f_new <= f_cur) and np.any(f_new < f_cur):
                        unique_results[idx] = (x_new, f_new)

        # Return the final set of points
        return unique_results
# EVOLVE-BLOCK-END
