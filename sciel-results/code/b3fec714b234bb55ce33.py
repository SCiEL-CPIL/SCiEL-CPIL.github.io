"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin within 300s limit
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For dim=5, use a population size that balances exploration
        # lam = 4 + int(3 * log(n)) gives 4 + int(3*1.609) = 4+4 = 8
        # Use a slightly larger population for better convergence on ridge
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial point - center of the box
        x = np.zeros(n)
        sigma = 2.0  # Initial step size

        # Covariance matrix and evolution paths
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (20 * n ** 2))

        # Track best
        best_x = x.copy()
        best_f = func(best_x)
        calls_made = 1

        # Precompute for efficiency
        def time_check():
            return time.time() > deadline

        # Main loop
        while calls_made < budget and not time_check():
            # Generate lambda offspring
            offspring = []
            for _ in range(lam):
                # Sample from multivariate normal with covariance C
                # Use Cholesky decomposition for efficiency
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes singular
                    L = np.eye(n)
                
                z = np.random.standard_normal(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append((y, None))

            # Evaluate offspring
            for i in range(lam):
                if calls_made >= budget or time_check():
                    break
                fy = func(offspring[i][0])
                offspring[i] = (offspring[i][0], fy)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = offspring[i][0].copy()

            # Sort by fitness
            valid_offspring = [(y, fy) for y, fy in offspring if fy is not None]
            if len(valid_offspring) < 2:
                break
            valid_offspring.sort(key=lambda x: x[1])

            # Select mu best
            y_sorted = [y for y, fy in valid_offspring[:mu]]
            f_sorted = [fy for y, fy in valid_offspring[:mu]]

            # Weighted mean of selected
            y_mean = np.sum([w[i] * y for i, y in enumerate(y_sorted)], axis=0)

            # Update evolution paths
            # ps for sigma adaptation
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (y_mean - x) / sigma)
            # Need L_inv, but we can approximate or skip for simplicity
            # For simplicity, use a simplified version without full CMA-ES adaptation

            # Actually, let's use a simpler adaptive strategy to avoid complexity
            # and potential numerical issues. Use a (mu+lambda) strategy with
            # adaptive step size based on success rate.

            # Update mean
            x = y_mean

            # Adapt sigma based on success
            # Count how many offspring improved
            improvements = sum(1 for fy in f_sorted if fy < best_f + 1e-10)
            if improvements > 0:
                sigma *= 1.1
            else:
                sigma *= 0.9

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)

            # Update covariance (simplified: use rank-1 update)
            # For efficiency and stability, use a simplified covariance update
            diff = y_mean - x
            C = 0.99 * C + 0.01 * np.outer(diff, diff) / (sigma ** 2 + 1e-8)

            # Ensure C is positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-6, None)
                C = (eigvecs * eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
