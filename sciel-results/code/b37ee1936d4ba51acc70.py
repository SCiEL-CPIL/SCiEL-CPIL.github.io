"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with parallel tempering (replica exchange) for global search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Number of replicas
        n_replicas = 10
        # Population size per replica
        pop_size = 4 + int(np.floor(3 * np.log(d)))  # standard CMA-ES lambda

        # Initialize replicas at different temperature scales
        # Each replica has: centroid, sigma, covariance matrix
        centroids = np.zeros((n_replicas, d))
        sigmas = np.zeros(n_replicas)
        covs = np.zeros((n_replicas, d, d))
        fvals = np.zeros(n_replicas)

        for i in range(n_replicas):
            # Different initial positions for diversity
            centroids[i] = rng.uniform(-4, 4, d)
            # Different sigma scales (temperature)
            sigmas[i] = 1.0 * (1.5 ** i)  # geometric temperature ladder
            covs[i] = np.eye(d)
            fvals[i] = func(centroids[i])

        best_x = centroids[np.argmin(fvals)].copy()
        best_f = np.min(fvals)

        # CMA-ES parameters
        mu = pop_size // 2  # number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        d_amp = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        # Learning rates
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Evolve strategies
        ps = np.zeros((n_replicas, d))
        pc = np.zeros((n_replicas, d))

        # Swap interval
        swap_interval = 10

        call_count = n_replicas  # initial evaluations

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        generation = 0
        while call_count < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population for each replica
            offspring = np.zeros((n_replicas, pop_size, d))
            offspring_f = np.zeros((n_replicas, pop_size))

            for r in range(n_replicas):
                # Generate offspring
                for j in range(pop_size):
                    if call_count >= budget:
                        break
                    z = rng.standard_normal(d)
                    x = centroids[r] + sigmas[r] * (covs[r] @ z)
                    x = np.clip(x, -5, 5)
                    offspring[r, j] = x
                    offspring_f[r, j] = func(x)
                    call_count += 1

                # Sort offspring by fitness
                order = np.argsort(offspring_f[r, :])
                # Select top mu
                selected = offspring[r, order[:mu]]
                selected_f = offspring_f[r, order[:mu]]

                # Compute weighted mean
                mean_new = np.sum(weights[:, None] * selected, axis=0)

                # Update covariance matrix
                y = (selected - centroids[r]) / sigmas[r]
                delta_sigma = (np.linalg.norm(mean_new - centroids[r]) / sigmas[r] -
                               np.sqrt(1 - np.exp(-2 * min(1, c_sigma) * mu_eff)))
                ps[r] = (1 - c_sigma) * ps[r] + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                        (mean_new - centroids[r]) / sigmas[r]
                h_sigma = 1 if np.linalg.norm(ps[r]) / np.sqrt(1 - (1 - c_sigma) ** 2) < \
                           (1 + 2 / (d + 1)) * np.sqrt((mu_eff - 1) / (d + 1)) else 0
                pc[r] = (1 - c_c) * pc[r] + h_sigma * c_c * (mean_new - centroids[r]) / sigmas[r]
                rank_1 = np.outer(pc[r], pc[r])
                rank_mu = np.sum(weights[:, None, None] * np.outer(y[:, k], y[:, k])
                                 for k in range(mu)) if mu > 0 else np.zeros((d, d))
                # Simplified rank_mu
                rank_mu = np.zeros((d, d))
                for k in range(mu):
                    rank_mu += weights[k] * np.outer(y[k], y[k])
                covs[r] = (1 + c_1 * (d_amp - 1) - c_1 - c_mu) * covs[r] + \
                          c_1 * (rank_1 - (1 - h_sigma) * (1 - c_c) * c_c * np.outer(pc[r], pc[r])) + \
                          c_mu * rank_mu

                # Update sigma
                sigmas[r] *= np.exp(c_sigma / d_amp * (np.linalg.norm(ps[r]) /
                                                       np.sqrt(1 - (1 - c_sigma) ** 2) -
                                                       np.sqrt((mu_eff - 1) / (d + 1))))
                sigmas[r] = np.clip(sigmas[r], 1e-10, 10.0)

                # Update centroid
                centroids[r] = mean_new
                fvals[r] = np.min(selected_f)

                # Update global best
                if fvals[r] < global_best_f:
                    global_best_f = fvals[r]
                    global_best_x = centroids[r].copy()

            generation += 1

            # Parallel tempering swap
            if generation % swap_interval == 0:
                # Sort replicas by fitness
                order = np.argsort(fvals)
                best_idx = order[0]
                worst_idx = order[-1]

                # Swap centroids and sigmas between best and worst
                if fvals[best_idx] < fvals[worst_idx]:
                    # Best replica helps worst: share some information
                    # Blend centroids
                    alpha = 0.3
                    centroids[worst_idx] = (1 - alpha) * centroids[worst_idx] + \
                                           alpha * centroids[best_idx]
                    # Blend sigmas (geometric mean)
                    sigmas[worst_idx] = np.sqrt(sigmas[worst_idx] * sigmas[best_idx])
                    # Re-evaluate worst replica's new position
                    fvals[worst_idx] = func(centroids[worst_idx])
                    call_count += 1
                    if fvals[worst_idx] < global_best_f:
                        global_best_f = fvals[worst_idx]
                        global_best_x = centroids[worst_idx].copy()

                # Also do a full swap between best and worst for diversity
                if fvals[order[0]] < fvals[order[-1]]:
                    # Swap sigma scales to allow exploration
                    temp_sigma = sigmas[order[0]].copy()
                    sigmas[order[0]] = sigmas[order[-1]]
                    sigmas[order[-1]] = temp_sigma

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
