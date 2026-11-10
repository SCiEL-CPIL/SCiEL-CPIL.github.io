"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class BudgetExceeded(Exception):
    pass


class Optimizer:
    """Multi-objective optimizer for RE21 using multi-start augmented epsilon-constraint
    with L-BFGS-B, Nelder-Mead polish, and hypervolume-guided refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill
        rng = np.random.default_rng(12345)
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

        # Phase 1: Sobol-like quasi-random initial sampling
        n_init = 30
        # Use a simple quasi-random sequence (Halton-like) for diversity
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            # Halton sequence in base (d+2)
            vals = []
            for i in range(1, n_init + 1):
                x = 0.0
                p = d + 2
                n = i
                while n > 0:
                    x += (n % p) / p
                    n //= p
                vals.append(x)
            X_init[:, d] = vals
        X_init = lo + span * X_init
        F_init = []
        for x in X_init:
            try:
                F_init.append(evaluate(x))
            except BudgetExceeded:
                break
        if not F_init:
            return mid
        F_init = np.array(F_init)

        archive_X = list(X_init[:len(F_init)])
        archive_F = list(F_init)

        f1_min = F_init[:, 0].min()
        f1_max = F_init[:, 0].max()
        f2_min = F_init[:, 1].min()
        f2_max = F_init[:, 1].max()

        # Phase 2: Find individual optima with multi-start
        x_opt1 = None
        x_opt2 = None
        best_f1 = np.inf
        best_f2 = np.inf

        # Try multiple starting points for f1 optimization
        start_points_f1 = [mid]
        # Add points from initial sampling that have good f1
        idx_f1 = np.argsort(F_init[:, 0])[:5]
        start_points_f1.extend([X_init[i] for i in idx_f1])

        for sp in start_points_f1:
            if not check_time() or used >= self.budget - 40:
                break
            try:
                def obj_f1(x):
                    return evaluate(x)[0]
                res1 = minimize(obj_f1, np.clip(sp, lo, hi), method='L-BFGS-B',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 10, 'ftol': 1e-10, 'gtol': 1e-8})
                if res1.success:
                    x_candidate = np.clip(res1.x, lo, hi)
                    f_candidate = evaluate(x_candidate)
                    if f_candidate[0] < best_f1:
                        best_f1 = f_candidate[0]
                        x_opt1 = x_candidate
                        archive_X.append(x_candidate)
                        archive_F.append(f_candidate)
                        f1_min = min(f1_min, f_candidate[0])
            except BudgetExceeded:
                break
            except Exception:
                pass

        # Try multiple starting points for f2 optimization
        start_points_f2 = [mid]
        idx_f2 = np.argsort(F_init[:, 1])[:5]
        start_points_f2.extend([X_init[i] for i in idx_f2])

        for sp in start_points_f2:
            if not check_time() or used >= self.budget - 40:
                break
            try:
                def obj_f2(x):
                    return evaluate(x)[1]
                res2 = minimize(obj_f2, np.clip(sp, lo, hi), method='L-BFGS-B',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 10, 'ftol': 1e-10, 'gtol': 1e-8})
                if res2.success:
                    x_candidate = np.clip(res2.x, lo, hi)
                    f_candidate = evaluate(x_candidate)
                    if f_candidate[1] < best_f2:
                        best_f2 = f_candidate[1]
                        x_opt2 = x_candidate
                        archive_X.append(x_candidate)
                        archive_F.append(f_candidate)
                        f2_min = min(f2_min, f_candidate[1])
            except BudgetExceeded:
                break
            except Exception:
                pass

        # Phase 3: Multi-start augmented epsilon-constraint method
        # Use 50 diverse starting points with dynamically adjusted epsilon constraints
        n_eps = 15
        if used > self.budget - 100:
            n_eps = max(8, n_eps - 4)
        if used > self.budget - 60:
            n_eps = max(5, n_eps - 4)
        if used > self.budget - 30:
            n_eps = max(3, n_eps - 3)

        # Epsilon values for f2 constraint
        eps_values = np.linspace(f2_max * 1.05, f2_min * 0.995, n_eps)
        eps_values = np.sort(eps_values)[::-1]  # descending

        # Create diverse starting points for each epsilon
        n_starts_per_eps = 3
        all_starts = []
        for i in range(n_eps):
            for j in range(n_starts_per_eps):
                if x_opt1 is not None and x_opt2 is not None:
                    w = i / max(1, n_eps - 1)
                    base = w * x_opt1 + (1 - w) * x_opt2
                else:
                    base = mid
                # Add small perturbation for diversity
                perturbation = rng.normal(0, 0.1 * span, size=self.dim)
                x0 = np.clip(base + perturbation, lo, hi)
                all_starts.append((i, x0))

        # Sort starts by epsilon value to process in order
        all_starts.sort(key=lambda t: eps_values[t[0]])

        for i, x0 in all_starts:
            if not check_time() or used >= self.budget - 10:
                break
            eps = eps_values[i]
            x0 = np.clip(x0, lo, hi)

            # Augmented epsilon-constraint: minimize f1 + mu*(f2 - eps)^+
            mu = 100.0

            def objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = evaluate(x_clipped)
                f1, f2 = f[0], f[1]
                penalty = mu * max(0.0, f2 - eps) ** 2
                return f1 + penalty

            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  bounds=list(zip(lo, hi)),
                                  options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-10})
                if result.success:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = evaluate(x_best)
                    archive_X.append(x_best)
                    archive_F.append(f_best)
            except BudgetExceeded:
                break
            except Exception:
                pass

        # Phase 4: Hypervolume-guided Nelder-Mead polish on non-dominated points
        if check_time() and used < self.budget - 5 and archive_F:
            F_arr = np.array(archive_F)
            n = len(F_arr)
            # Find non-dominated points
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom[i] = False
                        break

            nd_indices = np.where(nondom)[0]
            max_polish = min(8, len(nd_indices))
            if max_polish > 0:
                # Sort by hypervolume contribution proxy
                ref = np.array([3000.0, 0.06])
                hv_contrib = np.zeros(len(nd_indices))
                for k, idx in enumerate(nd_indices):
                    # Points with lower f1 and f2 contribute more to hypervolume
                    hv_contrib[k] = -(F_arr[idx, 0] + F_arr[idx, 1] * 50000)
                top_idx = nd_indices[np.argsort(hv_contrib)[:max_polish]]

                for idx in top_idx:
                    if not check_time() or used >= self.budget - 3:
                        break
                    x_start = np.clip(archive_X[idx], lo, hi)
                    try:
                        def nm_obj(x):
                            x_c = np.clip(x, lo, hi)
                            f = evaluate(x_c)
                            return f[0] + f[1] * 50000
                        nm_res = minimize(nm_obj, x_start, method='Nelder-Mead',
                                          options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12})
                        if nm_res.success:
                            x_nm = np.clip(nm_res.x, lo, hi)
                            f_nm = evaluate(x_nm)
                            archive_X.append(x_nm)
                            archive_F.append(f_nm)
                    except BudgetExceeded:
                        break
                    except Exception:
                        pass

        # Phase 5: Final high-precision polish on best points
        if check_time() and used < self.budget - 2 and archive_F:
            F_arr = np.array(archive_F)
            n = len(F_arr)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom[i] = False
                        break

            nd_indices = np.where(nondom)[0]
            # Polish top 3 with very tight tolerance
            if len(nd_indices) > 0:
                top3 = nd_indices[np.argsort(-(F_arr[nd_indices, 0] + F_arr[nd_indices, 1] * 50000))[:3]]
                for idx in top3:
                    if not check_time() or used >= self.budget - 1:
                        break
                    x_start = np.clip(archive_X[idx], lo, hi)
                    try:
                        def final_obj(x):
                            x_c = np.clip(x, lo, hi)
                            f = evaluate(x_c)
                            return f[0] + f[1] * 50000
                        final_res = minimize(final_obj, x_start, method='L-BFGS-B',
                                             bounds=list(zip(lo, hi)),
                                             options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-12})
                        if final_res.success:
                            x_final = np.clip(final_res.x, lo, hi)
                            f_final = evaluate(x_final)
                            archive_X.append(x_final)
                            archive_F.append(f_final)
                    except BudgetExceeded:
                        break
                    except Exception:
                        pass

        # Phase 6: Final cleanup - return first archive point
        return archive_X[0] if archive_X else mid
# EVOLVE-BLOCK-END
