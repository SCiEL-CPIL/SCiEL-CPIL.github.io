import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for BBOB f16 (Weierstrass)."""

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

        # Phase 1: CMA-ES with restarts
        # CMA-ES adapts the covariance matrix to handle rotated/anisotropic landscapes
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_ // 2
        # Weights: log-weighted
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)

        # Step size parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Restart schedule: each restart gets a portion of the budget
        num_restarts = 5
        restart_budget = max(30, (cma_limit - 10) // num_restarts)

        for restart_idx in range(num_restarts):
            if not time_ok() or calls >= cma_limit:
                break

            # Initialize mean and sigma for this restart
            if best_x is None:
                mean = rng.uniform(-4.5, 4.5, d)
            else:
                # Perturb from best with decreasing spread
                spread = 2.0 / (restart_idx + 1)
                mean = np.clip(best_x + spread * rng.standard_normal(d), -5.0, 5.0)

            sigma = 1.5 / (restart_idx + 1)
            C = np.eye(d)
            C_sqrt_inv = np.eye(d)
            ps = np.zeros(d)
            pc = np.zeros(d)
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            h_sigma = chi_d * (1 + c_sigma) / (1 - c_sigma)

            # Initial population
            pops = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5.0, 5.0)
                pops.append(y)

            # Evaluate initial population
            fs = []
            for y in pops:
                fy = float(func(y))
                calls += 1
                fs.append(fy)
                record(y, fy)

            # Sort by fitness
            order = np.argsort(fs)
            pops_sorted = [pops[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update best
            if fs_sorted[0] < best_fx:
                best_fx = fs_sorted[0]
                best_x = pops_sorted[0].copy()

            no_improve = 0
            stag_limit = 30

            while time_ok() and calls < cma_limit:
                # Generate population
                pops = []
                for i in range(lambda_):
                    z = rng.standard_normal(d)
                    y = mean + sigma * (C_sqrt_inv @ z)
                    y = np.clip(y, -5.0, 5.0)
                    pops.append(y)

                # Evaluate population
                fs = []
                for y in pops:
                    fy = float(func(y))
                    calls += 1
                    fs.append(fy)
                    record(y, fy)

                # Sort by fitness
                order = np.argsort(fs)
                pops_sorted = [pops[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Update best
                if fs_sorted[0] < best_fx:
                    best_fx = fs_sorted[0]
                    best_x = pops_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                new_mean = np.zeros(d)
                for i in range(mu):
                    new_mean += w[i] * pops_sorted[i]
                mean = new_mean

                # Update sigma (CSA)
                z_mean = np.zeros(d)
                for i in range(mu):
                    ydiff = pops_sorted[i] - mean
                    z = C_sqrt_inv @ ydiff / sigma
                    z_mean += w[i] * z

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (z_mean / np.linalg.norm(z_mean + 1e-12) * chi_d if np.linalg.norm(z_mean) > 1e-12 else 0)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
                sigma = float(np.clip(sigma, 1e-8, 5.0))

                # Update covariance matrix
                delta_h = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * h_sigma else 0
                pc = (1 - c_1) * pc + delta_h * np.sqrt(c_1 * (2 - c_1) * mu_eff) * (mean - new_mean) / sigma
                C = (1 + c_1 * (mu_eff - 2) - c_1 - c_mu) * C
                C += c_1 * np.outer(pc, pc)
                for i in range(mu):
                    ydiff = (pops_sorted[i] - new_mean) / sigma
                    C += c_mu * w[i] * np.outer(ydiff, ydiff)

                # Renormalize C to maintain determinism
                det_C = np.linalg.det(C)
                if det_C > 1e-12:
                    C *= (1.0 / det_C) ** (1.0 / d)

                # Update C_sqrt_inv using eigen-decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    C_sqrt_inv = np.eye(d)

                if no_improve > stag_limit:
                    break

        # Phase 2: local polish with multiple candidates
        if best_x is not None and time_ok() and calls < budget:
            # Try polishing from best and a few nearby points
            candidates = [best_x.copy()]
            for _ in range(4):
                cand = np.clip(best_x + 0.3 * rng.standard_normal(d), -5.0, 5.0)
                candidates.append(cand)

            for x in candidates:
                if not time_ok() or calls >= budget:
                    break
                fx = float(func(x))
                calls += 1
                record(x, fx)

                # Coordinate descent with adaptive step
                for _ in range(2):
                    if not time_ok() or calls >= budget:
                        break
                    for coord in range(d):
                        if not time_ok() or calls >= budget:
                            break
                        for direction in (1, -1):
                            if not time_ok() or calls >= budget:
                                break
                            step = 0.2
                            for _ in range(8):
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

                # Adaptive random fine steps
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
                        sigma *= 1.08
                    else:
                        sigma *= 0.92
                    if sigma < 1e-5:
                        break

        # Safety: ensure a valid result
        if best_x is None:
            x = np.zeros(d)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
