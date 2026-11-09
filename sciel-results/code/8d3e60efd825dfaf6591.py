"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Deterministic LHS + Nelder-Mead refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        eval_count = 0

        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            eval_count += 1
            return x, f

        # Phase 1: Latin Hypercube Sampling for diverse initialization
        n_lhs = int(budget * 0.5)
        n_lhs = min(n_lhs, budget)

        sampler = qmc.LatinHypercube(d=dim, seed=42)
        sample = sampler.random(n=n_lhs)
        X_lhs = lo + sample * (hi - lo)

        X_all = []
        F_all = []

        for x in X_lhs:
            if time.time() > deadline:
                break
            result = safe_eval(x)
            if result is None:
                break
            X_all.append(result[0])
            F_all.append(result[1])

        if len(F_all) == 0:
            return np.array([])

        X_all = np.array(X_all)
        F_all = np.array(F_all)

        # Helper: Get non-dominated indices
        def get_nd_indices(F):
            n = len(F)
            if n == 0:
                return []
            nd_idx = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)
            return nd_idx

        # Phase 2: Nelder-Mead refinement on non-dominated points
        # Use remaining budget for local search
        remaining = budget - eval_count

        if remaining > 0:
            nd_idx = get_nd_indices(F_all)

            if nd_idx:
                # Select top non-dominated points for refinement
                F_nd = F_all[nd_idx]
                # Score by normalized sum (lower is better)
                F_max = np.max(F_all, axis=0) + 1e-10
                scores = np.sum(F_nd / F_max, axis=1)
                # Pick up to 5 best non-dominated points
                k = min(5, len(scores))
                top_k = np.argsort(scores)[:k]
                top_indices = [nd_idx[i] for i in top_k]

                # Each Nelder-Mead call uses up to ~2*dim evaluations
                # Budget per NM: remaining / len(top_indices), capped
                nm_budget_per = max(10, int(remaining / len(top_indices)))
                nm_budget_per = min(nm_budget_per, 2 * dim + 10)

                for idx in top_indices:
                    if time.time() > deadline:
                        break
                    if eval_count >= budget:
                        break

                    x0 = X_all[idx]

                    # Scalarized objective: weighted sum of objectives
                    # Use adaptive weights based on current Pareto front
                    weights = np.ones(n_obj) / n_obj

                    def scalarized(x):
                        if time.time() > deadline:
                            return np.inf
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        f = np.asarray(f, dtype=float)
                        eval_count += 1
                        # Record this point
                        X_all.append(x)
                        F_all.append(f)
                        return np.dot(weights, f)

                    # Run Nelder-Mead
                    try:
                        res = minimize(
                            scalarized,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': nm_budget_per, 'xatol': 1e-6, 'fatol': 1e-6}
                        )
                        if time.time() > deadline:
                            break
                        # Record the best point found
                        if eval_count < budget:
                            x_best = np.clip(res.x, lo, hi)
                            f_best = problem(x_best)
                            f_best = np.asarray(f_best, dtype=float)
                            eval_count += 1
                            X_all.append(x_best)
                            F_all.append(f_best)
                    except Exception:
                        pass

        if len(F_all) == 0:
            return np.array([])

        return np.array(X_all)

# EVOLVE-BLOCK-END
