"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        D = self.dim
        budget = self.budget

        # Track best
        best_x = np.zeros(D)
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of selected individuals
        # Weights: proportional to log rank
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = (2 - mueff / D + cc) / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs

        # Initial mean and sigma
        x_mean = np.zeros(D)
        sigma = 4.0  # Cover the box [-5, 5]

        # State variables
        pc = np.zeros(D)
        ps = np.zeros(D)
        C = np.eye(D)
        # Precompute sqrt of C for sampling
        def chol_update(C, delta, pc, cs, c1, cmu, weights, x_mean, y_sorted, sigma):
            """Update C and return new sqrt decomposition."""
            C += c1 * (np.outer(pc, pc) + (1 - (1 + 2 * cs) * cc) * C - C)
            # Actually standard update:
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 + 2 * cs) * cc) * C)
            for i in range(mu):
                C += (cmu * weights[i]) * np.outer(y_sorted[i], y_sorted[i])
            # Symmetrize
            C = 0.5 * (C + C.T)
            return C

        # Use eigendecomposition for sampling since scipy may not be needed
        # Actually let's use a simpler approach: sample using Cholesky or eig

        def sample_from_C(C, sigma, rng):
            """Sample from N(0, C) scaled by sigma."""
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            z = rng.standard_normal(D)
            return sigma * (eigvecs @ (np.sqrt(eigvals) * z))

        rng = np.random.RandomState(42)

        try:
            while call_count < budget and time.time() < deadline:
                # Generate population
                population = []
                for i in range(lam):
                    y = sample_from_C(C, sigma, rng)
                    x = x_mean + y
                    population.append((x, y))

                # Evaluate
                fitness = []
                for x, y in population:
                    f = eval_func(x)
                    fitness.append(f)

                # Sort by fitness
                indices = np.argsort(fitness)
                fitness_sorted = [fitness[i] for i in indices]
                y_sorted = [population[i][1] for i in indices]
                x_sorted = [population[i][0] for i in indices]

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean_new = np.zeros(D)
                for i in range(mu):
                    x_mean_new += weights[i] * x_sorted[i]
                
                # Update step size (CSA)
                # Use (x_mean_new - x_mean_old)
                ps = (1 - cs) * ps + cs * np.sqrt(cs / (1 - cs)) * (x_mean_new - x_mean_old) / sigma
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (D + 1)) * np.sqrt(D + 1) else 0
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(D) - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)
                
                # Update C
                # Use (x_mean_new - x_mean_old)
                pc = (1 - cc) * pc + cc * hsig * (x_mean_new - x_mean_old) / sigma
                
                # Standard CMA-ES covariance update
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 + 2 * cs) * cc) * C)
                for i in range(mu):
                    C += (cmu * weights[i]) * np.outer(y_sorted[i], y_sorted[i])
                C = 0.5 * (C + C.T)
                
                x_mean = x_mean_new

                # Check termination: if sigma is very small or no progress
                if sigma < 1e-8:
                    break

        except (TimeoutError, ValueError):
            pass
        except Exception:
            pass

        return (best_x, best_f)
# EVOLVE-BLOCK-END
