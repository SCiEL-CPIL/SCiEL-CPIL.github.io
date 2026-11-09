"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box is a safe start
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Step size
        sigma = 2.0
        # Learning rates
        cc = (4 + n / lam) / ((n + 4) * (2 + n / lam))
        cmu = min(1 - cc, 2 * (2 + np.sqrt((n - 2) / (lam + 2)) - 2) / ((n + 2) * (lam + 2) - 2))
        # Mean vector
        m = x.copy()
        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition cache
        D = np.ones(n)
        B = np.eye(n)
        # Path variables
        ps = np.zeros(n)
        hc = (1 - cc) ** 2  # threshold
        cs = (4 + n / lam) / ((n + 4) + 2 * max(0, cmu - 1))
        hs = (1 - cs) ** 2  # threshold

        calls_made = 1
        # We have budget - 1 calls left
        remaining = budget - 1

        # Precompute some constants
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        while remaining > 0:
            # Generate population
            # Use eigen decomposition to sample from N(0, C)
            # Update eigen decomposition if needed (do it every iteration for simplicity, n=5 is small)
            B, D = np.linalg.eigh(C)
            # D contains eigenvalues, we need sqrt for step
            sqrt_D = np.sqrt(D)

            # Generate offspring
            offspring = []
            offspring_vals = []
            for i in range(lam):
                # Sample z ~ N(0, I)
                z = rng.standard_normal(n)
                # Transform to N(0, C)
                y = B @ (sqrt_D * z)
                # New candidate
                x_new = m + sigma * y
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                # Evaluate
                if remaining >= lam:
                    # We can evaluate all at once
                    fx_new = func(x_new)
                    remaining -= 1
                    calls_made += 1
                else:
                    # Not enough budget for full population, break
                    break
                
                offspring.append(x_new)
                offspring_vals.append(fx_new)

            if not offspring:
                break

            # Sort offspring by fitness
            indices = np.argsort(offspring_vals)
            offspring = [offspring[i] for i in indices]
            offspring_vals = [offspring_vals[i] for i in indices]
            mu = min(lam // 2, len(offspring))
            
            # Update mean
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            m_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)

            # Update covariance matrix
            # Compute weighted sum of outer products
            diff = np.array([offspring[i] - m for i in range(mu)])
            C_new = C.copy()
            for i in range(mu):
                C_new += (weights[i] / sigma ** 2) * np.outer(diff[i], diff[i])

            # Rank-mu update (simplified: use rank-one update for ps)
            # Update ps
            inv_sqrt_C = B @ np.diag(1.0 / sqrt_D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sqrt_C @ (m_new - m)

            # Update C with rank-one update based on ps
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1 + 2 / (n + 1)) * chi_n)
            hs = 1.0 if hsig else 0.0

            C_new += ((cmu - cmu / (2 * mu_eff)) * hs * (n + 1 - 2 / mu_eff) + (2 * cmu - cmu / mu_eff) * (1 - hs)) * (
                np.outer(ps, ps) / (sigma ** 2)
            )

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            m = m_new

            # Update covariance
            # Ensure positive semi-definite
            C_new = (C_new + C_new.T) / 2
            # Regularize if needed
            C = C_new

            # Update best
            if offspring_vals[0] < best_fx:
                best_fx = offspring_vals[0]
                best_x = offspring[0].copy()

            # If we can't do a full generation, break
            if remaining < lam:
                # Use remaining budget for (1+1) style local search
                while remaining > 0:
                    # Simple hill climbing step
                    step = sigma * rng.standard_normal(n)
                    x_try = np.clip(m + step, -5, 5)
                    fx_try = func(x_try)
                    remaining -= 1
                    calls_made += 1
                    if fx_try < best_fx:
                        best_fx = fx_try
                        best_x = x_try.copy()
                        m = x_try.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
