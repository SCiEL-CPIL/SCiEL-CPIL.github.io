import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_fx = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_fx, calls
            if calls >= budget:
                return best_fx
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Scan with LHS
        n_scan = min(50, budget // 4)
        # Simple LHS generation
        n_bins = int(np.ceil(n_scan ** (1.0 / dim)))
        scan_points = []
        for i in range(n_scan):
            x = np.zeros(dim)
            for d in range(dim):
                bin_idx = (i + d * n_scan // n_bins) % n_bins
                frac = (bin_idx + rng.uniform()) / n_bins
                x[d] = frac * 10.0 - 5.0
            scan_points.append(x)

        for x in scan_points:
            if not time_ok() or calls >= budget:
                break
            fx = evaluate(x)

        if not time_ok() or calls >= budget:
            return best_x, best_fx

        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_chol = np.linalg.cholesky(C)
        C_chol_inv = np.linalg.inv(C_chol)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        no_improvement = 0
        stall_threshold = 5 * lambda_pop
        generation = 0

        while time_ok() and calls < budget:
            if budget - calls < lambda_pop:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = mean + sigma * C_chol @ z
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)

            # Evaluate
            offspring_fx = []
            for y in offspring:
                if not time_ok() or calls >= budget:
                    break
                fy = evaluate(y)
                offspring_fx.append((fy, y))

            if not offspring_fx:
                break

            # Sort
            offspring_fx.sort(key=lambda t: t[0])
            
            # Update best
            if offspring_fx[0][0] < best_fx - 1e-12:
                no_improvement = 0
            else:
                no_improvement += 1

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(offspring_fx))):
                new_mean += weights[i] * offspring_fx[i][1]

            # Update path variables
            C_sqrt_inv = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
            C_sqrt_inv = np.linalg.inv(C_sqrt_inv)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Update sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (generation + 1)) < (1.4 + 2 / (dim + 1)) * chi_N else 0.0
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_N))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance
            delta_sigma = h_sigma * (1 / c_sigma - 1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(min(mu, len(offspring_fx))):
                diff = offspring_fx[i][1] - mean
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c_1 * (delta_sigma - 1) + c_mu) * C - c_1 * rank_one - c_mu * rank_mu
            C = 0.5 * (C + C.T)

            # Update Cholesky
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_chol_inv = np.linalg.inv(C_chol)
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)
                C_chol_inv = np.eye(dim)

            mean = new_mean
            generation += 1

            # Stagnation Restart
            if no_improvement > stall_threshold:
                mean = best_x.copy()
                sigma = 0.5 * sigma
                C = np.eye(dim)
                C_chol = np.eye(dim)
                C_chol_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improvement = 0

        return best_x, best_fx
