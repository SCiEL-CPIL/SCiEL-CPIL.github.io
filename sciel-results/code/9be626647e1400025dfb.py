"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initial population: center + some spread
        x = np.zeros(dim)
        # Try a few random starts to find a good initial region
        best_x = None
        best_f = np.inf
        for _ in range(5):
            cand = rng.uniform(-4, 4, dim)
            f = func(cand)
            if f < best_f:
                best_f = f
                best_x = cand.copy()
        x = best_x.copy()
        fx = best_f

        # CMA-ES parameters
        lam = 10  # population size
        c_sigma = (lam + 2) / (dim + lam + 5)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lam))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        hsig = np.sqrt((1 - np.exp(-2 / (dim + lam))) * (dim + lam) * (2 - c_c))

        calls = 6  # initial evaluations

        while calls < budget:
            # Check deadline
            if time.time() > deadline - 2:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Apply evolution path
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

                if calls >= budget:
                    break
                if time.time() > deadline - 2:
                    break

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < fx:
                fx = offspring[0][1]
                x = offspring[0][0].copy()

            # Selection weights
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Mean of selected
            x_new = sum(weights[i] * offspring[i][0] for i in range(mu))

            # Evolution path for sigma
            diff = (x_new - x) / (sigma + 1e-16)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim) * diff
            hsig = np.sqrt((1 - np.exp(-2 / (dim + lam))) * (dim + lam) * (2 - c_c))

            # Evolution path for covariance
            if np.linalg.norm(ps) / chi < (1.4 + 1 / dim) * np.sqrt(1 - np.exp(-1 / (dim + lam / 2))):
                pass  # h = 1
            else:
                pass  # h = 0, but we'll just use a simplified version

            pc = (1 - c_c) * pc + c_c * diff

            # Covariance matrix update (simplified: diagonal + rank-1)
            # For simplicity, we'll use a simplified CMA-ES with diagonal covariance
            # Actually, let's do proper CMA-ES with full covariance

            # Full covariance update
            # H = sum of outer products
            H = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (offspring[i][0] - x) / (sigma + 1e-16)
                H += weights[i] * np.outer(diff_i, diff_i)

            # Update B and D
            # This is expensive, so let's use a simplified approach
            # Actually, for dim=5, full covariance is fine
            C = np.eye(dim)  # We'll maintain C directly
            # But we started with B and D... let's switch to maintaining C directly

            # Let me restart with a cleaner CMA-ES implementation
            break

        # Fallback: simple (1+1)-ES if CMA-ES didn't complete
        # Actually, let's just do a proper implementation from scratch

        return x, fx


# Let me rewrite this properly
class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        rng = np.random.default_rng(42)

        # Initial point
        x0 = rng.uniform(-3, 3, dim)
        fx0 = func(x0)
        x = x0.copy()
        fx = fx0
        calls = 1

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (lam + 2) / (dim + lam + 5)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lam))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        sigma = 1.5
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        def Binv(v):
            return B.T @ v

        while calls < budget:
            if time.time() > deadline - 2:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))
                if calls >= budget or time.time() > deadline - 2:
                    break

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < fx:
                fx = offspring[0][1]
                x = offspring[0][0].copy()

            # Mean of selected
            x_new = sum(weights[i] * offspring[i][0] for i in range(mu))

            # Delta
            diff = (x_new - x) / (sigma + 1e-16)

            # Evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim) * diff

            # Evolution path for covariance
            pc = (1 - c_c) * pc + c_c * diff

            # Covariance matrix update
            H = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (offspring[i][0] - x) / (sigma + 1e-16)
                H += weights[i] * np.outer(diff_i, diff_i)

            C = (1 + c_1 + c_mu) * C - c_1 * np.outer(pc, pc) - c_mu * H

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Step size adaptation
            sigma *= np.exp(c_sigma / chi * (np.linalg.norm(ps) / chi - 1))

            x = x_new

        return x, fx
# EVOLVE-BLOCK-END
