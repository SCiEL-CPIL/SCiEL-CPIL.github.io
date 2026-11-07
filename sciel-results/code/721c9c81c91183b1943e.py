import numpy as np
import time


class Optimizer:
    """CMA-ES (covariance-matrix adaptation) with multi-start seeding and local polish for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 0: quick multi-start to seed a rough best (helps AOCC early progress)
        # Minimizer is in [-4, 4]^5, so sample a slightly tighter box.
        for _ in range(min(10, budget)):
            if not time_ok() or calls >= budget:
                break
            x0 = rng.uniform(-4.5, 4.5, d)
            fx0 = float(func(x0))
            calls += 1
            record(x0, fx0)

        # Reserve budget for final polish
        polish_budget = max(50, int(0.15 * budget))
        cma_limit = budget - polish_budget

        # Phase 1: CMA-ES (structural change: covariance-matrix adaptation to the
        # rotated/anisotropic Weierstrass landscape, vs. axis-aligned (1+1)-ES).
        if best_x is not None and time_ok() and calls < cma_limit:
            # CMA-ES hyperparameters
            pop = max(4, int(4 + 3 * np.log(d)))
            mu = pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            nu = 2.0 / (weights[0] + weights[1] ** 2 - 1.0)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = (2 - mu_eff / d + c_c) / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            x_mean = best_x.copy()
            sigma = 1.0
            C = np.eye(d)
            C_inv = np.eye(d)
            # Cholesky of C (lower)
            C_chol = np.linalg.cholesky(C)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            no_improve = 0
            stag_limit = 3 * (d + 1) * pop

            while time_ok() and calls < cma_limit:
                # Generate population
                samples = []
                for i in range(pop):
                    if not time_ok() or calls >= cma_limit:
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x_mean + sigma * (C_chol @ z), -5.0, 5.0)
                    fy = float(func(y))
                    calls += 1
                    samples.append((fy, y))
                if len(samples) < pop:
                    break

                samples.sort(key=lambda t: t[0])
                ys = [s[1] for s in samples]
                # Update best
                for fy, y in samples:
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                # Weighted mean of mu best
                y_w = np.zeros(d)
                for i in range(mu):
                    y_w += weights[i] * ys[i]
                x_mean_old = x_mean.copy()
                x_mean = y_w

                # Update C inverse via Sherman-Morrison-Woodbury on C
                # p_sigma update
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_chol.T @ ((y_w - x_mean_old) / sigma))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
                sigma = float(np.clip(sigma, 1e-9, 10.0))

                # p_c update
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * ((y_w - x_mean_old) / sigma)

                # Covariance update: C = (1 - c_1 - c_mu + h_sigma * c_1) * C + c_1 * p_c p_c^T + c_mu * sum w_i y_i y_i^T
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls / pop)))) < (1.4 + 2 / (d + 1)) * chi_d else 0.0
                C = (1 - c_1 - c_mu + h_sigma * c_1) * C + c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(ys[i] / sigma, ys[i] / sigma)
                # Symmetrize
                C = 0.5 * (C + C.T)

                # Re-cholesky (with jitter for numerical stability)
                for _ in range(10):
                    try:
                        C_chol = np.linalg.cholesky(C + 1e-9 * np.eye(d))
                        break
                    except np.linalg.LinAlgError:
                        C = 0.5 * C + 0.5 * np.eye(d) * np.trace(C) / d

                if no_improve > stag_limit:
                    break

        # Phase 2: local polish (coordinate descent + adaptive random fine steps)
        if best_x is not None and time_ok() and calls < budget:
            x = best_x.copy()
            fx = best_fx
            # coordinate descent with adaptive step
            for _ in range(3):
                if not time_ok() or calls >= budget:
                    break
                for coord in range(d):
                    if not time_ok() or calls >= budget:
                        break
                    for direction in (1, -1):
                        if not time_ok() or calls >= budget:
                            break
                        step = 0.3
                        for _ in range(10):
                            if not time_ok() or calls >= budget:
                                break
                            y = x.copy()
                            y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                            fy = float(func(y))
                            calls += 1
                            if fy < fx:
                                x = y
                                fx = fy
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                step *= 0.7
                            else:
                                break
            # adaptive random fine steps
            sigma = 0.05
            while time_ok() and calls < budget:
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        # Safety: ensure a valid result
        if best_x is None:
            x = np.zeros(d)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
