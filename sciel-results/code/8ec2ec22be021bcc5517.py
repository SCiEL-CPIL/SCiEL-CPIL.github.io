# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Multi-start CMA-ES with adaptive sigma and deadline guard.
    Uses a population-based evolutionary strategy (CMA-ES variant) for better
    exploration than (1+1)-ES, with multiple restarts to handle the BBOB f14
    landscape's multiple local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def remaining_calls():
            return self.budget - calls_made

        def time_left():
            return deadline - time.time()

        def eval_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            if time_left() < 0.5:
                return None
            fx = float(func(x))
            calls_made += 1
            return fx

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * math.log(self.dim)))  # population size
        # Standard CMA-ES initialization
        # We'll do multiple restarts

        n_restarts = 6
        calls_per_restart = max(10, (self.budget - 5) // n_restarts)

        for restart in range(n_restarts):
            if calls_made >= self.budget or time_left() < 1.0:
                break

            # Initialize center: for first restart use origin-ish, for others use best found
            if restart == 0:
                c = np.zeros(self.dim)
            else:
                c = best_x.copy() if best_x is not None else rng.uniform(-3, 3, self.dim)

            # Initial sigma: start moderate, adapt
            sigma = 2.5
            # Covariance matrix: identity
            C = np.eye(self.dim)
            # Eigenvalues of C
            eigval, eigvec = np.linalg.eigh(C)
            # Step sizes
            step_sizes = np.sqrt(eigval)
            # Cumulation path
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            # Adaptation parameters
            cc = (4 + lambda_pop / self.dim) / (self.dim + 4 + 2 * lambda_pop / self.dim)
            cs = (lambda_pop + 2) / (self.dim + lambda_pop + 5)
            cmu = min(0.3, (2 - cc + cs) / (self.dim + 2 + cc + cs) - 1)
            ccov = (5 + cmu - cc) / (self.dim + 5)
            # Chi-squared expected value
            chi_lambda = np.sqrt(lambda_pop) * (1 - 1/(4*lambda_pop) + 1/(32*lambda_pop**2))

            # Evaluate initial population
            population = []
            for i in range(lambda_pop):
                if calls_made >= self.budget or time_left() < 0.5:
                    break
                # Generate candidate
                z = rng.standard_normal(self.dim)
                # Transform using current C and step sizes
                # x = c + sigma * sum_j z_j * eigvec_j * step_size_j
                # Simpler: use Cholesky or just diagonal for speed in low dim
                # For dim=5, we can do full eigendecomposition
                try:
                    # Use eigenvector representation
                    y = np.array([z[j] * step_sizes[j] for j in range(self.dim)])
                    x_candidate = c + sigma * (eigvec @ y)
                    x_candidate = np.clip(x_candidate, -5, 5)
                    fx = eval_func(x_candidate)
                    if fx is not None:
                        population.append((x_candidate, fx, z))
                except Exception:
                    continue

            if len(population) < 2:
                continue

            # Sort by fitness
            population.sort(key=lambda t: t[1])

            # Update best
            if population[0][1] < best_f:
                best_f = population[0][1]
                best_x = population[0][0].copy()

            # Run CMA-ES iterations
            max_iter = max(1, (calls_per_restart - len(population)) // lambda_pop)
            for it in range(max_iter):
                if calls_made >= self.budget or time_left() < 0.5:
                    break

                # Select top mu individuals
                mu = lambda_pop // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)

                # Weighted mean
                wmean = np.zeros(self.dim)
                for i in range(mu):
                    wmean += weights[i] * population[i][0]
                wmean /= np.sum(weights[:mu])

                # Update covariance matrix
                old_c = c.copy()
                c = wmean

                # Cumulation
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / sigma * (eigvec.T @ (c - old_c))
                hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1.4 + 2 / (self.dim + 1)) * chi_lambda
                pc = (1 - cc) * pc + hsig * cc / sigma * (c - old_c)

                # Update C
                rank_mu = np.zeros((self.dim, self.dim))
                for i in range(mu):
                    diff = (population[i][0] - old_c) / sigma
                    rank_mu += weights[i] * np.outer(diff, diff)
                rank_one = np.outer(pc, pc)
                delta_diag = (1 - hsig) * cc * (2 - cc)
                C = (1 + ccov * (cmu * (rank_mu - C) + delta_diag * rank_one) - ccov) * C
                # Ensure positive definiteness
                C = (C + C.T) / 2

                # Recompute eigenvalues
                try:
                    eigval, eigvec = np.linalg.eigh(C)
                    eigval = np.maximum(eigval, 1e-12)
                    step_sizes = np.sqrt(eigval)
                except Exception:
                    C = np.eye(self.dim)
                    eigval, eigvec = np.linalg.eigh(C)
                    step_sizes = np.sqrt(eigval)

                # Adapt sigma
                sigma *= math.exp(cs / chi_lambda * (np.linalg.norm(ps) / chi_lambda - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)

                # Generate new population
                new_population = []
                for i in range(lambda_pop):
                    if calls_made >= self.budget or time_left() < 0.5:
                        break
                    try:
                        z = rng.standard_normal(self.dim)
                        y = np.array([z[j] * step_sizes[j] for j in range(self.dim)])
                        x_candidate = c + sigma * (eigvec @ y)
                        x_candidate = np.clip(x_candidate, -5, 5)
                        fx = eval_func(x_candidate)
                        if fx is not None:
                            new_population.append((x_candidate, fx, z))
                    except Exception:
                        continue

                if len(new_population) < 2:
                    break

                new_population.sort(key=lambda t: t[1])
                population = new_population

                # Update best
                if population[0][1] < best_f:
                    best_f = population[0][1]
                    best_x = population[0][0].copy()

        # Final polish: local refinement around best point
        if best_x is not None and calls_made < self.budget and time_left() > 1.0:
            local_sigma = 0.1
            for _ in range(min(50, self.budget - calls_made)):
                if calls_made >= self.budget or time_left() < 0.5:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)
                if fy is not None:
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        local_sigma *= 1.3
                    else:
                        local_sigma *= 0.8

        return best_x, best_f

# EVOLVE-BLOCK-END
