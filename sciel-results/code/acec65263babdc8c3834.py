import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
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
        es_limit = budget - polish_budget

        # Population size for CMA-ES
        lambda_pop = 10  # population size

        # CMA-ES parameters
        c1 = 1 / (4 * d + 4)
        cmu = 2 / ((d + 1.3)**2 + 4)
        c_sigma = (d + 1.8) / (d + 9)
        d_sigma = 1 + 2 * max(0, np.sqrt((d - 1) / (d + 1)) - 1) + c_sigma
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        # Restart schedule: (initial_sigma, spread_for_restart)
        schedule = [
            (1.5, 2.5),
            (1.2, 2.0),
            (0.9, 1.8),
            (0.6, 1.5),
            (0.4, 1.2),
            (0.25, 0.9),
            (0.15, 0.6),
            (0.08, 0.4),
            (0.04, 0.25),
            (0.02, 0.15),
            (0.01, 0.1),
        ]

        for sigma0, spread in schedule:
            if not time_ok() or calls >= es_limit:
                break

            # Initialize population center
            if best_x is None:
                x_mean = rng.uniform(-4.5, 4.5, d)
            else:
                x_mean = np.clip(best_x + spread * 0.5 * rng.standard_normal(d), -5.0, 5.0)

            # Initialize CMA-ES state
            sigma = sigma0
            C = np.eye(d)  # covariance matrix
            p_sigma = np.zeros(d)  # evolution path for sigma
            p_c = np.zeros(d)  # evolution path for covariance
            h_sigma = chi_d * (1 + c_sigma) / (1 - c_sigma)

            # Weighting for the best half of the population
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            no_improve = 0
            stag_limit = 50  # number of iterations without improvement before restart

            while time_ok() and calls < es_limit:
                # Generate population
                ys = []
                for _ in range(lambda_pop):
                    z = rng.standard_normal(d)
                    y = x_mean + sigma * (C @ z)
                    y = np.clip(y, -5.0, 5.0)
                    ys.append(y)

                # Evaluate population
                fs = []
                for y in ys:
                    fy = float(func(y))
                    calls += 1
                    fs.append(fy)
                    record(y, fy)

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = [ys[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Update best
                if fs_sorted[0] < best_fx:
                    best_fx = fs_sorted[0]
                    best_x = ys_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                x_mean_new = np.zeros(d)
                for i in range(mu):
                    x_mean_new += weights[i] * ys_sorted[i]
                x_mean = x_mean_new

                # Update sigma (CSA)
                z_mean = np.zeros(d)
                for i in range(mu):
                    z_mean += weights[i] * (C @ (ys_sorted[i] - x_mean) / sigma)
                # Note: this is an approximation; proper CMA-ES uses the inverse square root of C
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (z_mean / np.linalg.norm(z_mean + 1e-12) * chi_d if np.linalg.norm(z_mean) > 1e-12 else 0)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
                sigma = float(np.clip(sigma, 1e-8, 5.0))

                # Update covariance matrix
                delta_h = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * h_sigma else 0
                p_c = (1 - c1) * p_c + delta_h * np.sqrt(c1 * (2 - c1) * mu_eff) * (x_mean - x_mean_new) / sigma
                C = (1 + c1 * (mu_eff - 2) - c1 - cmu) * C
                C += c1 * np.outer(p_c, p_c)
                for i in range(mu):
                    ydiff = (ys_sorted[i] - x_mean_new) / sigma
                    C += cmu * weights[i] * np.outer(ydiff, ydiff)

                # Renormalize C to maintain determinism
                det_C = np.linalg.det(C)
                if det_C > 1e-12:
                    C *= (1.0 / det_C) ** (1.0 / d)

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
