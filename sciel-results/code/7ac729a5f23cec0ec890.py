"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
        mu = lam // 2  # number of selected parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial population size
        pop_size = lam

        # Initial center and sigma
        center = rng.uniform(-3, 3, dim)
        sigma = 1.5  # Initial step size

        # CMA-ES state
        ps = np.zeros(dim)  # evolution path for covariance adaptation
        hs = np.sqrt((1 + cc) * dim + 2)
        C = np.eye(dim)  # covariance matrix
        Cs = np.zeros((dim, dim))  # evolution path for covariance
        pc = np.zeros(dim)  # evolution path for CMA

        best_x = None
        best_f = float('inf')

        # Main loop
        while calls[0] < budget and time.time() < deadline:
            # Generate population
            if time.time() > deadline - 5:
                break

            # Sample population
            population = []
            for i in range(lam):
                # Whitening transformation
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                x = center + sigma * (B @ z)
                x = np.clip(x, -5.0, 5.0)
                population.append(x)

            # Evaluate population
            fvals = np.array([safe_call(x) for x in population])

            # Sort by fitness
            order = np.argsort(fvals)
            population = [population[i] for i in order]
            fvals = fvals[order]

            # Update best
            for x, f in zip(population, fvals):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Selection
            w = weights[:mu]
            xw = np.array(population[:mu])
            new_center = np.sum(w[:, np.newaxis] * xw, axis=0)

            # Update sigma (CSA - Cumulative Step Adaptation)
            old_sigma = sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (new_center - center))
            if np.linalg.norm(ps) / chi < (1.4 + 1 / dim) * hs:
                Cs = (1 - cs) * Cs + cs * (ps @ ps.T - np.eye(dim))

            # Update C (covariance matrix adaptation)
            diff = (new_center - center) / sigma
            pc = (1 - cc) * pc + cc * diff
            delta_hsig = (1 - np.exp(-cs * (np.linalg.norm(ps) / chi - 1))) if np.linalg.norm(ps) / chi > (1.4 + 1 / dim) * hs else 0
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * np.eye(dim)) + cmu * np.sum(w[:, np.newaxis, np.newaxis] * (xw - center)[:, np.newaxis, :] * (xw - center)[:, np.newaxis, :], axis=0) / (sigma ** 2)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization for numerical stability
            C += 1e-10 * np.eye(dim)

            # Update center and sigma
            center = new_center
            sigma = sigma * np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))

            # Adapt sigma based on population spread
            if sigma < 1e-6:
                break

            # Check convergence
            if np.std(fvals[:mu]) < 1e-10 and np.linalg.norm(new_center - center) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
