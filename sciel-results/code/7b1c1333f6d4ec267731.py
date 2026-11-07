"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 30  # population size
        mu = lam // 2  # number of recombination weights
        # Logarithmic linear weights
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)

        # Step size control
        cs = 1 / (5 * dim + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # Covariance matrix adaptation
        cmac = 1 + 2 * (mueff - 2) / (dim + 2)**2 - cs
        c1 = (mueff + 2) / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Inverse square root of C
        def invsqrtm(C):
            D2, B2 = np.linalg.eigh(C)
            D2 = np.clip(D2, 1e-20, None)
            D2inv = 1.0 / np.sqrt(D2)
            return B2 @ (D2inv[:, None] * B2.T)

        CS = invsqrtm(C)

        # Initial population
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Generate initial population
        def generate_population():
            nonlocal CS, sigma
            z = rng.standard_normal((lam, dim))
            return x + sigma * (z @ CS.T)

        pop = generate_population()
        for i in range(lam):
            if calls >= budget:
                break
            xi = np.clip(pop[i], -5, 5)
            fi = func(xi)
            calls += 1
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Sort population by fitness
        if calls < budget:
            fvals = np.array([func(np.clip(pop[i], -5, 5)) if i >= lam else 0 for i in range(lam)])
            # Actually we already evaluated them, let's restructure

        # Let me restructure the whole thing more carefully

        # Reset and do it properly
        x = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        CS = invsqrtm(C)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            z = rng.standard_normal((lam, dim))
            pop = x + sigma * (z @ CS.T)

            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                xi = np.clip(pop[i], -5, 5)
                fvals[i] = func(xi)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = xi.copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            y = pop[order[:mu]]
            fvals_sorted = fvals[order]

            # Update mean
            x_new = np.sum(w[:, None] * y, axis=0)

            # Update evolution paths
            pc = (1 - cs) * pc + cs / np.sqrt(cs * (2 - cs)) * (CS @ (x_new - x) / sigma)
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cs)**((calls + 1) / (cs * (dim + 1))))) < (1 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1))
            ps = (1 - ds) * ps + float(hsig) * ds / np.sqrt(ds * (2 - ds)) * (x_new - x) / sigma

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Update covariance matrix
            delta = float(hsig) * (1 - cs) * cs
            rank1 = np.outer(pc, pc)
            diff = y - x
            rankmu = np.sum(w[:, None, None] * (diff @ diff.T), axis=0)
            C = (1 + cmac * (c1 + cmu - 1) - c1 - cmu) * C + c1 * (rank1 + delta) + cmu * rankmu

            # Symmetrize
            C = (C + C.T) / 2

            # Update eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.clip(D, 1e-20, None)
            D = np.sqrt(D)
            CS = B @ (1.0 / D[:, None] * B.T)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
