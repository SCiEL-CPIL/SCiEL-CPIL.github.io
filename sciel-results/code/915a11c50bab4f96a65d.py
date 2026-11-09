"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class BudgetExceeded(Exception):
    pass


class Optimizer:
    """Multi-objective optimizer using Augmented Epsilon-Constraint Method with L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        used = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                raise BudgetExceeded()
            if not check_time():
                raise BudgetExceeded()
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            return f

        # Phase 1: Initial sampling to get bounds on objectives
        n_init = 20
        X_init = lo + span * rng.random((n_init, self.dim))
        F_init = []
        for x in X_init:
            try:
                F_init.append(evaluate(x))
            except BudgetExceeded:
                break
        if not F_init:
            return mid
        F_init = np.array(F_init)

        # Get objective bounds from initial samples
        f1_min, f1_max = F_init[:, 0].min(), F_init[:, 0].max()
        f2_min, f2_max = F_init[:, 1].min(), F_init[:, 1].max()

        archive_X = []
        archive_F = []

        # Add initial points to archive
        for i in range(len(F_init)):
            archive_X.append(X_init[i])
            archive_F.append(F_init[i])

        # Find individual optima to get tighter bounds
        # We will use a small budget for these
        # Minimize f1
        try:
            def obj_f1(x):
                return evaluate(x)[0]
            
            res1 = minimize(obj_f1, mid, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                            options={'maxiter': 5, 'ftol': 1e-6, 'gtol': 1e-4})
            if res1.success:
                x_opt1 = np.clip(res1.x, lo, hi)
                f_opt1 = evaluate(x_opt1)
                archive_X.append(x_opt1)
                archive_F.append(f_opt1)
                f1_min = min(f1_min, f_opt1[0])
        except BudgetExceeded:
            pass

        # Minimize f2
        try:
            def obj_f2(x):
                return evaluate(x)[1]
            
            res2 = minimize(obj_f2, mid, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                            options={'maxiter': 5, 'ftol': 1e-6, 'gtol': 1e-4})
            if res2.success:
                x_opt2 = np.clip(res2.x, lo, hi)
                f_opt2 = evaluate(x_opt2)
                archive_X.append(x_opt2)
                archive_F.append(f_opt2)
                f2_min = min(f2_min, f_opt2[1])
        except BudgetExceeded:
            pass

        # Phase 2: Epsilon-Constraint Method to trace Pareto front
        # Limit number of sub-problems to ensure budget safety
        n_eps = 5
        if used > self.budget - 50:
            n_eps = 2
        
        eps_values = np.linspace(f2_max * 1.05, f2_min * 1.0, n_eps)
        eps_values = np.sort(eps_values)[::-1]

        # Use weighted sum as fallback for starting points
        weights = np.linspace(1.0, 0.0, n_eps)

        for i, (eps, w) in enumerate(zip(eps_values, weights)):
            if not check_time() or used >= self.budget - 10:
                break

            # Starting point: blend between the two individual optima if available, else mid
            if 'res1' in locals() and 'res2' in locals() and res1.success and res2.success:
                x0 = w * np.clip(res1.x, lo, hi) + (1 - w) * np.clip(res2.x, lo, hi)
            else:
                x0 = mid
            x0 = np.clip(x0, lo, hi)

            # Augmented epsilon-constraint: minimize f1 + mu*(f2 - eps)^+
            mu = 10.0

            def objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = evaluate(x_clipped)
                f1, f2 = f[0], f[1]
                penalty = mu * max(0, f2 - eps) ** 2
                return f1 + penalty

            # Try L-BFGS-B with limited iterations
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  bounds=list(zip(lo, hi)),
                                  options={'maxiter': 10, 'ftol': 1e-8, 'gtol': 1e-6})
                if result.success:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = evaluate(x_best)
                    archive_X.append(x_best)
                    archive_F.append(f_best)
            except BudgetExceeded:
                break
            except Exception:
                pass

        # Phase 3: Local refinement around promising points
        if check_time() and used < self.budget - 10:
            # For each point in archive, try small perturbations
            for ax in archive_X[:]:
                if not check_time() or used >= self.budget - 2:
                    break
                for _ in range(3):
                    if not check_time() or used >= self.budget - 1:
                        break
                    perturbation = 0.05 * span * rng.standard_normal(self.dim)
                    x_try = np.clip(ax + perturbation, lo, hi)
                    try:
                        f_try = evaluate(x_try)
                        archive_X.append(x_try)
                        archive_F.append(f_try)
                    except BudgetExceeded:
                        break

        # Phase 4: Final cleanup - remove dominated points
        if archive_F:
            F_arr = np.array(archive_F)
            # Keep only non-dominated points
            nondom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if not nondom_mask[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j or not nondom_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom_mask[i] = False
                        break

        return archive_X[0] if archive_X else mid
# EVOLVE-BLOCK-END
