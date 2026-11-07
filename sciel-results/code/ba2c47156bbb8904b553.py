"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and covariance matrix."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial parameters
        x0 = rng.uniform(-3, 3, n)
        sigma = 1.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cc = (4 + n / 3) / (n + 4 + 2)
        cs = (n + 2) / (n + 6 + 2 * n / (n + 1))
        cmu = 2 / ((n + 4) ** 2 + 1)
        damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
        dampc = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + 2 * cc

        # Evaluation count tracking
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            eval_count += 1
            return func(x)

        # Initial evaluation
        x_best = x0.copy()
        f_best = evaluate(x_best)
        eval_count = 1

        # Population size
        lam = 4 + int(3 * n)
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # CMA-ES main loop
        while eval_count < budget:
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x_best + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                pop.append(y)

            # Evaluate population
            fvals = []
            for y in pop:
                fval = evaluate(y)
                fvals.append(fval)
                if eval_count >= budget:
                    break

            if len(fvals) < lam:
                # Pad with infinity for remaining
                fvals.extend([float('inf')] * (lam - len(fvals)))

            # Sort by fitness
            idx = np.argsort(fvals)
            pop_sorted = [pop[i] for i in idx[:mu]]
            fvals_sorted = [fvals[i] for i in idx[:mu]]

            # Update best
            if fvals_sorted[0] < f_best:
                x_best = pop_sorted[0]
                f_best = fvals_sorted[0]

            # Compute weighted mean
            m_new = np.zeros(n)
            for i in range(mu):
                m_new += weights[i] * pop_sorted[i]

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt @ (m_new - x_best) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - eval_count + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (n + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - x_best) / sigma

            # Update covariance matrix
            delta_c = (1 - hsig) * (1 - cc) * (2 - cc)
            C = (1 - cmu * delta_c) * C + cmu * sum(weights[i] * np.outer(pop_sorted[i] - x_best, pop_sorted[i] - x_best) / sigma ** 2 for i in range(mu))

            # Update step size (CSA-like)
            # Simple adaptation: increase if many improvements, decrease otherwise
            n_improvements = sum(1 for fv in fvals[:lam] if fv < f_best)
            if n_improvements > lam // 4:
                sigma *= 1.1
            elif n_improvements == 0:
                sigma *= 0.9
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Update x_best for next iteration (use weighted mean)
            x_best = m_new

            # Update C_sqrt via Cholesky decomposition
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add small diagonal
                C += 1e-6 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)

            if eval_count >= budget:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
