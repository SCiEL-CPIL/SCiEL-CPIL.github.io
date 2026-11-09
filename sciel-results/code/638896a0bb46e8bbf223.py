"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Constrained multi-objective search using Augmented Epsilon-Constraint Method."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        eval_count = [0]
        all_points = []
        all_objectives = []

        def count_problem(x):
            eval_count[0] += 1
            x = np.clip(x, lo, hi)
            f = problem(x)
            return np.array(f, dtype=float)

        def add_point(x, f):
            all_points.append(np.array(x, dtype=float).copy())
            all_objectives.append(np.array(f, dtype=float).copy())

        # Phase 1: Latin Hypercube sampling for diverse initialization
        n_init = min(40, self.budget // 8)
        rng = np.random.default_rng(0)
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X_init[:, d] = (perm + rng.random(n_init)) / n_init
        X_init = lo + (hi - lo) * X_init

        for x in X_init:
            f = count_problem(x)
            add_point(x, f)

        F_init = np.array(all_objectives)

        # Identify nadir and ideal points
        f1_min = F_init[:, 1].min()
        f1_max = F_init[:, 1].max()
        f0_min = F_init[:, 0].min()
        f0_max = F_init[:, 0].max()

        # Phase 2: Augmented Epsilon-Constraint Method
        # Use more epsilon values to better trace the Pareto front
        remaining = self.budget - eval_count[0]
        n_eps = min(15, max(10, remaining // 20))

        # Epsilon values: linearly spaced from f1_max to f1_min
        # This ensures we explore the full range of the second objective
        epsilons = np.linspace(f1_max, f1_min, n_eps)

        # For each epsilon, minimize f0 subject to f1 <= eps
        # Using augmented penalty: f0 + mu * max(0, f1 - eps)
        for i, eps in enumerate(epsilons):
            if eval_count[0] >= self.budget:
                break

            # Choose starting point: prefer points with f1 close to eps
            if len(all_objectives) > 0:
                F_curr = np.array(all_objectives)
                # Points that satisfy constraint (f1 <= eps) are preferred
                # Among those, prefer lower f0; among violators, prefer closest to eps
                mask_sat = F_curr[:, 1] <= eps + 1e-6
                if mask_sat.sum() > 0:
                    candidates = np.where(mask_sat)[0]
                    # Among satisfying points, pick one with lowest f0
                    best_sat = candidates[np.argmin(F_curr[candidates, 0])]
                    x0 = all_points[best_sat].copy()
                else:
                    # No point satisfies constraint; pick closest to eps
                    dists = np.abs(F_curr[:, 1] - eps)
                    x0 = all_points[np.argmin(dists)].copy()
            else:
                x0 = X_init[0].copy()

            # Use augmented epsilon constraint with linear penalty
            # This is more stable than quadratic for L-BFGS-B
            mu = 10.0  # Penalty weight

            def augmented_objective(x):
                f = count_problem(x)
                penalty = mu * max(0.0, f[1] - eps)
                return f[0] + penalty

            # Set maxiter based on remaining budget
            budget_left = self.budget - eval_count[0]
            maxiter = max(10, min(80, budget_left // (2 * dim)))
            maxls = min(20, max(5, budget_left // (4 * dim)))

            try:
                result = minimize(augmented_objective, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': maxiter, 'maxls': maxls, 'ftol': 1e-10, 'gtol': 1e-8})

                if result.fun < float('inf') and np.all(np.isfinite(result.x)):
                    x_best = np.clip(result.x, lo, hi)
                    f_best = count_problem(x_best)
                    add_point(x_best, f_best)

                    # If constraint was significantly violated, retry with higher penalty
                    if f_best[1] > eps + 1e-4:
                        mu *= 3.0
                        budget_left = self.budget - eval_count[0]
                        if budget_left > 15:
                            maxiter2 = max(5, min(40, budget_left // (2 * dim)))
                            try:
                                result2 = minimize(augmented_objective, x_best, method='L-BFGS-B',
                                                   bounds=list(zip(lo, hi)),
                                                   options={'maxiter': maxiter2, 'maxls': 10, 'ftol': 1e-10, 'gtol': 1e-8})
                                if result2.fun < float('inf') and np.all(np.isfinite(result2.x)):
                                    x_best2 = np.clip(result2.x, lo, hi)
                                    f_best2 = count_problem(x_best2)
                                    add_point(x_best2, f_best2)
                            except Exception:
                                pass
            except Exception:
                pass

        # Phase 3: Local refinement of non-dominated points
        if eval_count[0] < self.budget:
            F_curr = np.array(all_objectives)
            n_pts = len(F_curr)
            nondom_idx = []
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_curr[j] <= F_curr[i]) and np.any(F_curr[j] < F_curr[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom_idx.append(i)

            # Refine up to 7 best non-dominated points
            budget_left = self.budget - eval_count[0]
            n_refine = min(7, len(nondom_idx), budget_left // 12)

            for idx in nondom_idx[:n_refine]:
                if eval_count[0] >= self.budget:
                    break
                x0 = all_points[idx].copy()

                def refine_objective(x):
                    f = count_problem(x)
                    return f[0] + f[1]

                budget_left = self.budget - eval_count[0]
                maxiter = max(5, min(40, budget_left // (2 * dim)))

                try:
                    result = minimize(refine_objective, x0, method='L-BFGS-B',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': maxiter, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-10})
                    if result.fun < float('inf') and np.all(np.isfinite(result.x)):
                        x_best = np.clip(result.x, lo, hi)
                        f_best = count_problem(x_best)
                        add_point(x_best, f_best)
                except Exception:
                    pass

        if all_points:
            return np.array(all_points)
        else:
            return X_init
# EVOLVE-BLOCK-END
