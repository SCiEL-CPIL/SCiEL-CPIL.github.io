"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a short random scan and a deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s kill

        # --- Phase 0: quick random scan to pick a good basin ---
        # f14 has many local minima; a few random starts help.
        scan_budget = min(20, budget // 10)
        x_best = None
        f_best = np.inf
        for i in range(scan_budget):
            x = np.random.uniform(-4.5, 4.5, dim)
            f = func(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()
            if time.time() > deadline:
                break

        if x_best is None:
            x_best = np.zeros(dim)
            f_best = func(x_best)

        # --- Phase 1: CMA-ES ---
        # Use a subset of remaining budget for CMA-ES
        used = scan_budget
        remaining = budget - used

        # CMA-ES parameters
        lambda_ = 4 + int(2 * np.log(dim))  # population size
        # cap lambda for small dim
        lambda_ = max(lambda_, 4)

        # Initial sigma: moderate spread
        sigma = 1.0

        # Mean
        mean = x_best.copy()
        best_x = x_best.copy()
        best_f = f_best

        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Step size control
        cs = 1 + 2 * min(1.0 / (lambda_ - 1), 2 * (dim - 1) / (lambda_ + 3))
        ds = 1 + 2 * min(1.0 / (lambda_ - 1), (dim + 1) / 2 - 1)
        cc = (4 + dim / lambda_) / ((dim + 4) + 2 * lambda_ / (dim - 1))
        cmu = min(1 - cc, 2 * ((dim + 1. / 3) - lambda_ - 2) / ((dim + 2.) * (lambda_ + 2.)))
        cmu = max(cmu, 0.0)

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi_lambda = np.sqrt(lambda_) * (1 + 1 / (4 * dim) - 1 / (24 * dim ** 2))

        count_evals = used
        no_improve = 0

        while count_evals < budget:
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            offspring = []
            fitness = []
            for i in range(lambda_):
                # Check deadline per individual
                if time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                f_y = func(y)
                count_evals += 1
                offspring.append(y)
                fitness.append(f_y)

            if len(offspring) == 0:
                break

            # Sort by fitness
            order = np.argsort(fitness)
            offspring = [offspring[i] for i in order]
            fitness = [fitness[i] for i in order]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = offspring[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Weighted recombination
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            y_weighted = np.zeros(dim)
            for i in range(mu):
                y_weighted += weights[i] * offspring[i]

            # Update path evolution strategy
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ y_weighted)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (count_evals - used + 1) / lambda_))
                    < (1 + 2 / (dim + 1)) * chi_lambda)
            hsig = float(hsig)

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_weighted / sigma

            # Update covariance
            delta_d = hsig * (1 + cc) - 1
            diff = np.zeros((dim, dim))
            for i in range(mu):
                diff += np.outer(y_weighted - offspring[i], y_weighted - offspring[i])
            C = (1 + cmu * (mueff - 1) - cc * delta_d) * C + cmu * diff + cc * delta_d * np.outer(pc, pc)

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Update mean
            mean = mean + sigma * y_weighted
            mean = np.clip(mean, -5, 5)

            # Update sigma (CSA)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_lambda - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 20.0)

            # Recompute matrix square root periodically (every 5 generations or if needed)
            gen = count_evals // lambda_
            if gen % 5 == 0 or count_evals - used < lambda_:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

            # Early stopping if sigma is very small
            if sigma < 1e-8:
                break

            # If no improvement for many generations, try a small restart
            if no_improve > 5 * lambda_ and count_evals + lambda_ < budget:
                # Small perturbation
                mean = best_x + 0.5 * np.random.standard_normal(dim)
                mean = np.clip(mean, -5, 5)
                sigma = 0.5
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
