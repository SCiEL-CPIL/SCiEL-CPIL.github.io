"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES variant with adaptive population size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial point
        x = np.zeros(n)
        fx = func(x)

        # CMA-ES parameters
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Adaptation parameters
        cmu = 1 / (4 + n / 2)
        c1 = (2 / ((n + 1.3) ** 2 + n)) * (1 - cmu)
        cmu2 = cmu
        cc = (4 + n / 4) / (n + 4 + 2 * n / (n + 1))
        cs = (2 + cmu / 2) / (n + 2 + cmu / 2)
        damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs

        # Population size schedule
        # Start with larger population, shrink as we converge
        lambda_init = max(4, int(4 + 3 * n))  # e.g., 19 for n=5
        lambda_min = max(2, n // 2 + 1)  # e.g., 3 for n=5

        calls_made = 1
        best_x = x.copy()
        best_f = fx

        # Track convergence
        f_history = [fx]

        while calls_made < budget:
            # Adaptive population size based on progress
            progress = min(1.0, calls_made / (budget * 0.7))
            # Shrink population as we progress
            lam = max(lambda_min, int(lambda_init * (1 - 0.7 * progress)))
            
            # Generate offspring
            offspring = []
            for _ in range(lam):
                # Sample from multivariate normal using C and sigma
                z = rng.standard_normal(n)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            off_fx = []
            for y in offspring:
                if calls_made >= budget:
                    break
                fy = func(y)
                off_fx.append(fy)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if not off_fx:
                break

            # Sort by fitness
            indices = np.argsort(off_fx)
            off_fx_sorted = [off_fx[i] for i in indices]
            offspring_sorted = [offspring[i] for i in indices]

            # Update best
            if off_fx_sorted[0] < fx:
                fx = off_fx_sorted[0]
                x = offspring_sorted[0].copy()

            # Select top mu individuals
            mu = max(1, lam // 2)
            weights = np.ones(mu)
            weights /= np.sum(weights)

            # Compute new mean
            x_new = sum(w * y for w, y in zip(weights, offspring_sorted[:mu]))

            # Update evolution paths
            # Standard deviation of selected points
            d = np.sqrt(np.sum(weights) - 1 / np.sum(weights ** 2))
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * np.sum(weights)) * np.linalg.solve(L.T, (x_new - x) / sigma)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - lambda_init + 1) / n)) <
                    (1 + 2 / (n + 1)) * np.sqrt(np.trace(C)))
            pc = (1 - cc) * pc + float(hsig) * cc * (x_new - x) / sigma

            # Rank-mu update of C
            v = np.array([y - x for y in offspring_sorted[:mu]])
            C = (1 - c1 - cmu2) * C + c1 * (np.outer(pc, pc) + float(hsig) * (1 - float(hsig)) * C)
            C += cmu2 * np.sum(weights[:, None] * v @ v.T, axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(n)

            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 10)

            # Update mean
            x = x_new

            f_history.append(fx)

            # Early stopping if very converged
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
