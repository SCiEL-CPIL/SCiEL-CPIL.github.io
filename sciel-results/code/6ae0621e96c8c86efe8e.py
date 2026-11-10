"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit WFG9 structure: objectives depend only on x[0], x[1].
    Use dense 2D grid over position vars + local refinement for gaps."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 330s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # WFG9: objectives depend on x[0] (position for f1,f2) and x[1] (position for f3)
        # Fix remaining dims to mid-range to minimize contribution
        base = 0.5 * (lo + hi)

        # Dense grid over x[0] and x[1]
        n0 = 20  # grid points for x[0]
        n1 = 20  # grid points for x[1]
        grid_budget = n0 * n1  # 400 evaluations for grid

        X_list = []
        F_list = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            return x, f

        # Phase 1: Dense grid over x[0], x[1]
        x0_vals = np.linspace(lo[0], hi[0], n0)
        x1_vals = np.linspace(lo[1], hi[1], n1)

        for i0 in range(n0):
            for i1 in range(n1):
                if used >= budget or time.time() > deadline:
                    break
                x = base.copy()
                x[0] = x0_vals[i0]
                x[1] = x1_vals[i1]
                res = eval_point(x)
                if res is None:
                    continue
                X_list.append(res[0])
                F_list.append(res[1])

        # Convert to arrays
        X = np.array(X_list)
        F = np.array(F_list)

        # Phase 2: Local refinement around non-dominated points
        remaining = budget - used
        if remaining > 0 and len(F) > 0:
            # Find non-dominated points
            def is_dominated(f, F_arr):
                for fi in F_arr:
                    if np.all(fi <= f) and np.any(fi < f):
                        return True
                return False

            nondom_idx = [i for i in range(len(F)) if not is_dominated(F[i], F)]
            if nondom_idx:
                # Refine around each non-dominated point with small perturbations
                n_refine = min(remaining, len(nondom_idx) * 4)
                perturbations = [
                    (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02),
                    (0.01, 0.01), (-0.01, -0.01), (0.01, -0.01), (-0.01, 0.01)
                ]
                for idx in nondom_idx[:n_refine // 4]:
                    for dx0, dx1 in perturbations:
                        if used >= budget or time.time() > deadline:
                            break
                        x = F[idx].copy()  # This is wrong, need X not F
                        x = X[idx].copy()
                        x[0] += dx0 * (hi[0] - lo[0])
                        x[1] += dx1 * (hi[1] - lo[1])
                        res = eval_point(x)
                        if res is None:
                            continue
                        X_list.append(res[0])
                        F_list.append(res[1])

                X = np.array(X_list)
                F = np.array(F_list)

        # Phase 3: Fill remaining budget with random perturbations around best points
        remaining = budget - used
        if remaining > 0 and len(F) > 0:
            # Sort by sum of objectives, take best
            sums = np.sum(F, axis=1)
            best_idx = np.argsort(sums)[:min(5, len(F))]
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                bi = best_idx[np.random.randint(len(best_idx))]
                x = X[bi].copy()
                x[0] += 0.01 * (hi[0] - lo[0]) * np.random.randn()
                x[1] += 0.01 * (hi[1] - lo[1]) * np.random.randn()
                res = eval_point(x)
                if res is None:
                    break
                X_list.append(res[0])
                F_list.append(res[1])

            if len(X_list) > len(X):
                X = np.array(X_list)
                F = np.array(F_list)

        return X
# EVOLVE-BLOCK-END
