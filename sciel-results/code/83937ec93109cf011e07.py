"""Multi-fidelity adaptive coordinate descent + CMA-ES + L-BFGS-B for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Halton init, adaptive coordinate descent, CMA-ES, L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        all_x = []
        all_f = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f, all_x, all_f
            if eval_count >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            f_val = float(func(x))
            all_x.append(x.copy())
            all_f.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)

        # --- Phase 1: Halton Initialization ---
        n_init = 30
        primes = [2, 3, 5, 7, 11]

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, n_init + 1):
            if eval_count >= budget - 150 or time.time() > deadline:
                break
            pt = np.array([halton(i, primes[d]) for d in range(dim)])
            pt = pt * 10.0 - 5.0
            wrapped_func(pt)

        if eval_count < budget - 150 and time.time() <= deadline:
            wrapped_func(np.zeros(dim))

        # --- Phase 2: Adaptive Coordinate Descent ---
        # For each coordinate, fit a quadratic from 3 points to find optimal step
        # This reduces effective dimensionality and adapts step sizes per axis

        def coordinate_descent_round(x_center, n_rounds=3):
            nonlocal eval_count, best_x, best_f
            x_curr = x_center.copy()
            f_curr = wrapped_func(x_curr)
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()

            for round_num in range(n_rounds):
                if eval_count >= budget - 100 or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if eval_count >= budget - 100 or time.time() > deadline:
                        break
                    # Try to find best step along dimension d using quadratic fit
                    # Start with a moderate step
                    step = 1.0 / (round_num + 1)
                    x_plus = x_curr.copy()
                    x_minus = x_curr.copy()
                    x_plus[d] = np.clip(x_plus[d] + step, -5.0, 5.0)
                    x_minus[d] = np.clip(x_minus[d] - step, -5.0, 5.0)

                    f_plus = wrapped_func(x_plus)
                    f_minus = wrapped_func(x_minus)

                    # Quadratic fit: f(x) = a*x^2 + b*x + c
                    # Points: (x_curr[d], f_curr), (x_plus[d], f_plus), (x_minus[d], f_minus)
                    x0 = x_curr[d]
                    x1 = x_plus[d]
                    x2 = x_minus[d]

                    if abs(x1 - x0) < 1e-10 or abs(x2 - x0) < 1e-10:
                        continue

                    # Solve for vertex of quadratic
                    # Using Lagrange interpolation
                    denom1 = (x1 - x0) * (x1 - x2)
                    denom2 = (x0 - x1) * (x0 - x2)
                    denom3 = (x2 - x0) * (x2 - x1)

                    if abs(denom1) < 1e-10 or abs(denom2) < 1e-10 or abs(denom3) < 1e-10:
                        continue

                    a = f_curr / denom1 + f_plus / denom2 + f_minus / denom3
                    b = -(x1 + x2) * f_curr / denom1 - (x0 + x2) * f_plus / denom2 - (x0 + x1) * f_minus / denom3

                    if abs(a) < 1e-15:
                        continue

                    x_opt = -b / (2 * a)
                    x_opt = np.clip(x_opt, -5.0, 5.0)

                    # Only move if it's within reasonable bounds
                    if abs(x_opt - x_curr[d]) > 10 * step:
                        x_opt = x_curr[d] + np.sign(x_opt - x_curr[d]) * 10 * step

                    x_new = x_curr.copy()
                    x_new[d] = x_opt
                    f_new = wrapped_func(x_new)

                    if f_new < f_curr - 1e-10:
                        x_curr = x_new
                        f_curr = f_new
                        improved = True
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()

                if not improved:
                    break

            return x_curr, f_curr

        # Run coordinate descent from best point
        if eval_count < budget - 100 and time.time() < deadline:
            cd_x, cd_f = coordinate_descent_round(best_x, n_rounds=3)
            if cd_f < best_f:
                best_f = cd_f
                best_x = cd_x.copy()

        # --- Phase 3: CMA-ES Global Search ---
        pop_size = 12
        sigma = 2.0
        x_mean = best_x.copy()
        C = np.eye(dim)
        L = np.linalg.cholesky(C)

        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        stagnation_counter = 0
        max_cma_evals = int(budget * 0.45)
        cma_evals = 0

        while eval_count < budget - 80 and cma_evals < max_cma_evals and time.time() < deadline:
            # Generate population
            xs = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (L @ z)
                x = np.clip(x, -5.0, 5.0)
                xs.append(x)

            # Evaluate
            fs = []
            for x in xs:
                if eval_count >= budget - 80:
                    fs.append(float('inf'))
                    continue
                f = wrapped_func(x)
                fs.append(f)
                cma_evals += 1

            if len(fs) != pop_size:
                break

            # Sort
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * xs_sorted[i]

            # Update sigma
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (C_inv_sqrt @ (x_new_mean - x_mean)) / sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (cma_evals // pop_size + 1))) < (1 + 2 / (dim + 1)) * chi_n)

            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma

            delta_h = 1 if h_sigma else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(w[i] * np.outer(xs_sorted[i] - x_mean, xs_sorted[i] - x_mean) for i in range(mu))
            C += 1e-10 * np.eye(dim)

            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = x_new_mean

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = 0.5 * C + 0.5 * np.eye(dim)
                L = np.linalg.cholesky(C)

            if stagnation_counter > 15:
                x_mean = best_x.copy()
                sigma = 1.5
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_counter = 0
                for _ in range(4):
                    if eval_count >= budget - 80:
                        break
                    pt = best_x + rng.normal(0, 1.5, dim)
                    pt = np.clip(pt, -5.0, 5.0)
                    wrapped_func(pt)

        # --- Phase 4: Coordinate Descent Refinement ---
        if eval_count < budget - 30 and time.time() < deadline:
            cd_x, cd_f = coordinate_descent_round(best_x, n_rounds=2)
            if cd_f < best_f:
                best_f = cd_f
                best_x = cd_x.copy()

        # --- Phase 5: L-BFGS-B Multi-start Refinement ---
        if best_x is not None and eval_count < budget - 10 and time.time() < deadline:
            candidates = [best_x.copy()]

            if all_x:
                idx_sorted = np.argsort(all_f)
                for idx in idx_sorted[:6]:
                    pt = all_x[idx].copy()
                    is_distinct = True
                    for c in candidates:
                        if np.linalg.norm(pt - c) < 0.15:
                            is_distinct = False
                            break
                    if is_distinct:
                        candidates.append(pt)
                        if len(candidates) >= 5:
                            break

            bounds = [(-5.0, 5.0)] * dim
            for cand in candidates:
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                try:
                    result = minimize(
                        wrapped_func,
                        cand,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 80, 'maxfun': 40, 'maxls': 20, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # --- Phase 6: Final local random search ---
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
