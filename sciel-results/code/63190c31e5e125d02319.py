import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 292.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(20240101)

        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf

            x = np.asarray(x, dtype=np.float64)
            if x.shape != (d,):
                x = x.reshape(d)
            x = np.clip(x, lb, ub)

            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf

            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---- Phase 1: Global sampling (LHS-style + random) ----
        # Use Latin Hypercube for better space-filling, plus some random
        n_lhs = 50
        n_random = 15
        candidates = []

        for i in range(n_lhs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            # LHS: stratified sampling
            x = np.zeros(d)
            for j in range(d):
                x[j] = -5.0 + 10.0 * (i + rng.random()) / n_lhs
            f = call_func(x)
            candidates.append((f, x.copy()))

        for i in range(n_random):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            candidates.append((f, x.copy()))

        candidates.sort(key=lambda item: item[0])

        # ---- Phase 2: Differential Evolution (global optimizer) ----
        # DE is robust to multimodality (Rastrigin, Gallagher) and works well in 5D
        remaining = self.budget - calls_made
        if remaining > 30:
            de_budget = min(remaining * 2, 150)  # DE can use many calls internally
            try:
                def de_func(x):
                    if calls_made >= self.budget or time.time() > deadline:
                        return np.inf
                    return call_func(np.array(x))

                # Use best candidate as initial population hint
                de_bounds = [(-5.0, 5.0)] * d
                de_res = differential_evolution(
                    de_func,
                    bounds=de_bounds,
                    maxiter=50,
                    popsize=10,
                    tol=1e-10,
                    mutation=(0.4, 1.0),
                    recombination=0.8,
                    seed=42,
                    polish=False,  # We'll do our own polish
                    init='latinhypercube',
                    x0=None,
                    updating='immediate',
                    callback=None,
                    workers=1,
                )
            except Exception:
                de_res = None

        # ---- Phase 3: Local refinement from top candidates ----
        # Sort candidates again (DE may have found better points)
        if de_res is not None and de_res.x is not None:
            try:
                f_de = call_func(de_res.x)
                candidates.append((f_de, de_res.x.copy()))
            except Exception:
                pass

        candidates.sort(key=lambda item: item[0])
        # Deduplicate: keep only distinct points (within tolerance)
        seen = []
        unique_candidates = []
        for f, x in candidates:
            if np.isinf(f):
                continue
            is_dup = False
            for sx in seen:
                if np.linalg.norm(x - sx) < 0.01:
                    is_dup = True
                    break
            if not is_dup:
                unique_candidates.append((f, x))
                seen.append(x)

        # Run local optimization from top unique candidates
        max_local = min(10, len(unique_candidates))
        for i in range(max_local):
            if calls_made >= self.budget or time.time() > deadline:
                break

            _, x_start = unique_candidates[i]
            remaining = self.budget - calls_made
            if remaining < 5:
                break

            # L-BFGS-B for smooth convergence
            lbfgs_budget = min(remaining * 2, 40)
            try:
                res_lbfgs = minimize(
                    lambda x: call_func(np.array(x)),
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxls': 50,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )
            except Exception:
                res_lbfgs = None

            x_polished = x_start
            if res_lbfgs is not None and res_lbfgs.x is not None:
                x_polished = res_lbfgs.x

            # Nelder-Mead polish for non-smooth functions
            if calls_made < self.budget and time.time() < deadline:
                nm_budget = min(self.budget - calls_made, 20)
                if nm_budget > 5:
                    try:
                        res_nm = minimize(
                            lambda x: call_func(np.array(x)),
                            x_polished,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 100,
                                'xatol': 1e-10,
                                'fatol': 1e-12,
                                'adaptive': True,
                            }
                        )
                        if res_nm.x is not None:
                            x_polished = res_nm.x
                    except Exception:
                        pass

        # ---- Phase 4: Final high-precision polish on global best ----
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made

            # Multiple passes with decreasing tolerances
            for pass_idx, (ftol, gtol, nm_xatol, nm_fatol) in enumerate([
                (1e-14, 1e-10, 1e-10, 1e-12),
                (1e-16, 1e-12, 1e-12, 1e-14),
                (1e-18, 1e-14, 1e-14, 1e-16),
            ]):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                rem = self.budget - calls_made
                if rem < 3:
                    break

                # L-BFGS-B with tight tolerance
                try:
                    minimize(
                        lambda x: call_func(np.array(x)),
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'ftol': ftol,
                            'gtol': gtol,
                        }
                    )
                except Exception:
                    pass

                # NM polish
                if calls_made < self.budget and time.time() < deadline:
                    rem = self.budget - calls_made
                    if rem > 3:
                        try:
                            minimize(
                                lambda x: call_func(np.array(x)),
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': 50,
                                    'xatol': nm_xatol,
                                    'fatol': nm_fatol,
                                    'adaptive': True,
                                }
                            )
                        except Exception:
                            pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
