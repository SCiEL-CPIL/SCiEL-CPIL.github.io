"""CMA-ES + (1+1)-ES hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid CMA-ES + (1+1)-ES for BBOB f11.
    CMA-ES finds the global basin; (1+1)-ES polishes on the flat Discus landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Initial exploration ---
        points = []
        values = []

        center = np.zeros(dim)
        fc = func(np.clip(center, -5, 5))
        points.append(center)
        values.append(fc)
        evals = 1

        # Explore along each axis in both directions
        for i in range(dim):
            for d in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1.0, -1.0]:
                    if evals >= budget:
                        break
                    p = np.zeros(dim)
                    p[i] = sign * d
                    fp = func(np.clip(p, -5, 5))
                    points.append(p)
                    values.append(fp)
                    evals += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break

        # Add random points for diversity
        for _ in range(min(20, budget - evals)):
            if evals >= budget:
                break
            p = rng.uniform(-4, 4, dim)
            fp = func(np.clip(p, -5, 5))
            points.append(p)
            values.append(fp)
            evals += 1

        if evals >= budget:
            best_idx = int(np.argmin(values))
            return points[best_idx], values[best_idx]

        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx], dtype=np.float64)
        best_f = values[best_idx]

        # --- Phase 2: CMA-ES Optimization ---
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_pop // 2
        sigma = 1.0

        cc = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        cs = (lambda_pop / 2 + 1) / (dim + lambda_pop / 2 + 5)
        cmu = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        damps = 1 + 2 * max(0, np.sqrt((lambda_pop / 2 - 1) / (dim + 1)) - 1) + cs

        x_mean = best_x.copy()
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        try:
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            C_sqrt = L.T
            C_inv_sqrt = L_inv
        except:
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        gen = 0

        # Reserve ~30% of budget for local polish
        polish_budget = max(50, int(budget * 0.3))
        cma_budget = budget - polish_budget

        while evals < cma_budget:
            if time.time() > deadline:
                break

            gen += 1

            offspring = []
            offspring_f = []

            for i in range(lambda_pop):
                if evals >= cma_budget:
                    break

                z = rng.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            off_order = np.argsort(offspring_f)
            sorted_offspring = [offspring[j] for j in off_order]
            sorted_offspring_f = [offspring_f[j] for j in off_order]

            if sorted_offspring_f[0] < best_f:
                best_f = sorted_offspring_f[0]
                best_x = sorted_offspring[0].copy()

            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_offspring[i]

            x_diff = x_new - x_mean
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (x_diff / sigma)

            hsig = np.sqrt((1 + (1 - 2 * cs) ** 2) / (1 - cs) - 1) * (2 - cs)
            hs = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (dim + 1)) * hsig else 0.0

            p_c = (1 - cc) * p_c + hs * np.sqrt(cc * (2 - cc) * mu_eff) * C_inv_sqrt @ (x_diff / sigma)

            delta_hsig = (1 - hs) * cc * (2 - cc)
            C = (1 + cmu * (mu_eff + 2) - cmu) * C + \
                cmu * np.outer(p_c, p_c) * (1 + delta_hsig - cc * (2 - cc)) + \
                cmu / mu_eff * sum(weights[i] * np.outer(sorted_offspring[i] - x_mean, sorted_offspring[i] - x_mean) for i in range(mu))

            C = (C + C.T) / 2 + 1e-12 * np.eye(dim)

            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * gen)) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x_mean = x_new

            if gen % 10 == 0:
                try:
                    L = np.linalg.cholesky(C)
                    L_inv = np.linalg.inv(L)
                    C_sqrt = L.T
                    C_inv_sqrt = L_inv
                except:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = max(sigma, 0.1)

        # --- Phase 3: (1+1)-ES Local Polish with adaptive sigma ---
        x_local = best_x.copy()
        f_local = best_f
        sigma_local = max(sigma * 0.1, 0.05)

        no_improve = 0
        max_no_improve = 50

        while evals < budget:
            if time.time() > deadline:
                break

            z = rng.standard_normal(dim)
            y = x_local + sigma_local * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            evals += 1

            if fy < f_local:
                x_local = y
                f_local = fy
                sigma_local = max(sigma_local * 1.1, 1e-8)
                no_improve = 0
            else:
                sigma_local = max(sigma_local * 0.9, 1e-8)
                no_improve += 1

            if no_improve >= max_no_improve:
                break

            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
