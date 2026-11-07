"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        # Use larger population for better exploration in low dimensions
        lam = max(4, int(4 + 3 * np.log(n)))
        # For n=5, this gives lam=9; increase to 12 for better performance
        lam = max(lam, 12)
        # Ensure we don't exceed budget
        if lam * 10 > self.budget:
            lam = max(2, self.budget // 10)

        # Initial point: center of the box
        x = np.zeros(n)
        # Initial step size - tuned for the problem scale
        sigma = 2.0

        # Initialize covariance matrix to identity
        C = np.eye(n)
        # Eigen-decomposition for sampling
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-10))

        # Cumulation paths
        pc = np.zeros(n)
        ps = np.zeros(n)
        # Weighting - use better default weights
        mu = lam // 2
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Better boundary handling: use penalty + reflection
        def handle_boundary(y):
            y = y.copy()
            for i in range(n):
                if y[i] < -5:
                    # Reflect with damping
                    y[i] = -5 + (-5 - y[i]) * 0.3
                elif y[i] > 5:
                    y[i] = 5 - (y[i] - 5) * 0.3
            return np.clip(y, -5, 5)

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        calls = 1
        no_improve = 0
        prev_best = best_f
        for gen in range(1000):
            if calls >= self.budget:
                break

            # Sample population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = handle_boundary(y)
                offspring.append(y)

            # Evaluate
            fvals = []
            for y in offspring:
                if calls >= self.budget:
                    break
                fy = func(y)
                fvals.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if len(fvals) < lam:
                break

            # Stagnation detection: restart with smaller sigma if stuck
            if no_improve > lam * 3 and sigma > 1e-4:
                sigma *= 0.5
                no_improve = 0

            # Sort by fitness
            order = np.argsort(fvals)
            offspring = [offspring[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update mean
            x_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)

            # Update covariance matrix
            # First, update cumulation paths
            try:
                y_inv = np.linalg.solve(C, (x_new - x) / sigma)
            except np.linalg.LinAlgError:
                y_inv = (x_new - x) / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_inv
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))
                    < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_inv

            # Update C
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                try:
                    zi = np.linalg.solve(C, (offspring[i] - x) / sigma)
                except np.linalg.LinAlgError:
                    zi = (offspring[i] - x) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c1 * delta_hsig + cmu - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += (1e-10 - np.min(eigvals) + 1e-10) * np.eye(n)

            # Update step size (CSA)
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            sigma = np.clip(sigma, 1e-8, 50)

            # Update mean
            x = x_new

            # Update eigen-decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))

        return best_x, best_f
# EVOLVE-BLOCK-END
