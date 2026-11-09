"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start: coarse sampling -> coordinate line search -> adaptive ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s/330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return calls_used < budget and time.time() <= deadline

        # ---- Phase 1: coarse random sampling ----
        n_coarse = min(24, max(8, budget // 12))
        coarse_x, coarse_f = [], []
        for _ in range(n_coarse):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if np.isfinite(fx):
                coarse_x.append(x)
                coarse_f.append(fx)

        if not coarse_x:
            x0 = np.zeros(dim)
            fx0 = safe_call(x0)
            update_best(x0, fx0 if np.isfinite(fx0) else 0.0)
            return best_x, best_f

        order = np.argsort(coarse_f)
        k = min(4, len(coarse_x))
        top = list(order[:k])

        # ---- Phase 2: coordinate line search (exploits separability) ----
        # Binary search on each coordinate line for a local minimum.
        def line_search(x, fx, budget_ls):
            nonlocal best_x, best_f
            for _ in range(budget_ls):
                if not time_ok():
                    break
                improved_any = False
                for d in range(dim):
                    if not time_ok():
                        break
                    base = x[d]
                    lo, hi = max(-5.0, base - 5.0), min(5.0, base + 5.0)
                    if hi - lo < 1e-4:
                        continue
                    # bracket a descent direction
                    step = 1.0
                    down = None
                    up = None
                    for _ in range(6):
                        if not time_ok():
                            break
                        cand_lo = max(-5.0, base - step)
                        cand_hi = min(5.0, base + step)
                        if abs(cand_lo - base) > 1e-9:
                            x[d] = cand_lo
                            flo = safe_call(x)
                            if flo < fx:
                                down = cand_lo
                        if abs(cand_hi - base) > 1e-9:
                            x[d] = cand_hi
                            fhi = safe_call(x)
                            if fhi < fx:
                                up = cand_hi
                        x[d] = base
                        if down is not None or up is not None:
                            break
                        step *= 0.5
                    if down is None and up is None:
                        continue
                    # binary search within the improving interval
                    if down is not None:
                        a, b = down, base
                    else:
                        a, b = base, up
                    for _ in range(7):
                        if not time_ok():
                            break
                        mid = 0.5 * (a + b)
                        x[d] = mid
                        fm = safe_call(x)
                        if fm < fx:
                            fx = fm
                            b = mid
                            improved_any = True
                        else:
                            a = mid
                    x[d] = b
                    update_best(x, fx)
                if not improved_any:
                    break
            return x, fx

        # ---- Phase 3: adaptive isotropic ES (handles non-separable funcs) ----
        def es_refine(x, fx, budget_es):
            nonlocal best_x, best_f
            sigma = 1.0
            for _ in range(budget_es):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if np.isfinite(fy) and fy < fx:
                    # adapt sigma from the step that succeeded (covariance-free)
                    delta = np.linalg.norm(y - x)
                    if delta > 1e-9:
                        sigma = np.clip(sigma * (0.8 + 0.6 * (delta / (sigma + 1e-9))),
                                        1e-6, 8.0)
                    x, fx = y, fy
                    update_best(x, fx)
                else:
                    sigma *= 0.85
                if sigma < 1e-7:
                    break
            return x, fx

        # Distribute remaining budget among top starts (better start -> more).
        remaining = budget - calls_used
        weights = np.array([1.0 / (i + 1) for i in range(k)], dtype=float)
        weights /= weights.sum()

        for i, idx in enumerate(top):
            if not time_ok():
                break
            x = coarse_x[idx].copy()
            fx = coarse_f[idx]
            alloc = int(remaining * weights[i])
            alloc = max(8, alloc)
            ls_budget = max(4, int(alloc * 0.45))
            es_budget = max(4, alloc - ls_budget)

            x, fx = line_search(x, fx, ls_budget)
            x, fx = es_refine(x, fx, es_budget)

        # ---- Phase 4: global restart pool to escape local minima ----
        if time_ok() and best_x is not None:
            n_restart = min(6, max(1, (budget - calls_used) // 25))
            for _ in range(n_restart):
                if not time_ok():
                    break
                # start near the global best with a moderate perturbation
                x = np.clip(best_x + 1.5 * rng.standard_normal(dim), -5, 5)
                fx = safe_call(x)
                if not np.isfinite(fx):
                    continue
                update_best(x, fx)
                x, fx = line_search(x, fx, max(3, (budget - calls_used) // (n_restart + 1) // 2))
                x, fx = es_refine(x, fx, max(3, (budget - calls_used) // (n_restart + 1) // 2))

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0
        return best_x, best_f
# EVOLVE-BLOCK-END
