"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB f12 (Bent Cigar) optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        N = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf

        calls_made = 0

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(d))  # 4 + 3*ln(5) ≈ 4 + 4.83 = 8
        # CMA-ES weights
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))

        # Initial state
        x = np.zeros(d)  # Start at center
        sigma = 2.0  # Initial step size
        C = np.eye(d)  # Covariance matrix
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Track fitness history for stagnation detection
        no_improve = 0
        stagnation_threshold = 50

        # Evaluate initial point
        x_init = np.zeros(d)
        fx_init = func(x_init)
        calls_made += 1
        best_x = x_init.copy()
        best_f = fx_init
        x = x_init.copy()

        # Main CMA-ES loop
        while calls_made < N:
            # Check time
            if time.time() > deadline:
                break

            # Check if we have enough budget for a full generation
            if calls_made + lambda_ > N:
                # Do individual evaluations until budget is exhausted
                while calls_made < N:
                    if time.time() > deadline:
                        break
                    y = x + sigma * rng.multivariate_normal(np.zeros(d), C)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                break

            # Generate population
            population = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.multivariate_normal(np.zeros(d), C)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                population.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(population) < lambda_:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[0])

            # Check for improvement
            if population[0][0] < best_f - 1e-10:
                no_improve = 0
            else:
                no_improve += 1

            # Stagnation handling: if stuck for too long, restart with new center
            if no_improve > stagnation_threshold:
                # Reset to a new random point near the current best
                x = best_x + rng.standard_normal(d) * 1.0
                x = np.clip(x, -5, 5)
                sigma = 1.5
                C = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                no_improve = 0
                continue

            # Update mean (weighted average of best mu solutions)
            w_mean = np.zeros(d)
            for i in range(mu):
                w_mean += weights[i] * population[i][1]
            x_new = w_mean

            # Update sigma (CSA)
            c_sigma = np.sqrt((1 + 2 * max(0, cs - 1) * cs / (cs + 1)) * mueff / (d + 2))
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff / lambda_) * (x_new - x) / sigma
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance matrix
            c_c = np.sqrt((1 + 2 * max(0, cc - 1) * cc / (cc + 1)) * mueff / (d + 2))
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mueff / lambda_) * (x_new - x) / sigma

            # Rank-one update
            rank_one = np.outer(p_c, p_c)
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n < (1.4 + 2 / (d + 1)) * (1 - c_sigma) else 0
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                z = (population[i][1] - x) / sigma
                rank_mu += weights[i] * np.outer(z, z)

            C = (1 + c1 * h_sigma * (cc - 1 + cc) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            C = (C + C.T) / 2  # Symmetrize

            # Update mean
            x = x_new

            # Ensure sigma doesn't go too small or too large
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 20.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
