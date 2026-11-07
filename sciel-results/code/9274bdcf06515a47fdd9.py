"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Best overall
        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        # We'll do a few restarts if time permits
        max_restarts = 5
        calls_used = 0

        for restart in range(max_restarts):
            # Initialize
            if restart == 0:
                # Try a few initial points to find a good starting region
                candidates = []
                for _ in range(5):
                    x0 = rng.uniform(-5, 5, dim)
                    f0 = func(x0)
                    calls_used += 1
                    candidates.append((f0, x0))
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                candidates.sort()
                x = candidates[0][1].copy()
                fx = candidates[0][0]
            else:
                # Restart from best found with some perturbation
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1, dim), -5, 5)
                    fx = func(x)
                    calls_used += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_used += 1
                    best_f = fx
                    best_x = x.copy()

            if calls_used >= budget:
                break

            # CMA-ES-like parameters
            sigma = 2.0
            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            # We'll use a (lambda/2 + 1, lambda)-ES scheme
            mu = lambda_ // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)
            # Covariance adaptation
            c1 = (2 / ((dim + 1.3)**2 + mueff))
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
            cs = (mueff + 2) / (dim + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            # State
            C = np.eye(dim)
            C_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            # Precompute sqrt of C
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            sqrtC = B @ np.diag(D) @ B.T
            inv_sqrtC = B @ np.diag(1.0/D) @ B.T

            # Track best within this run
            run_best_f = fx
            run_best_x = x.copy()

            # Main loop
            while calls_used < budget:
                if time.time() > deadline - 5:
                    break

                # Generate lambda_ offspring
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (sqrtC @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    offspring.append((fy, y))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    if fy < run_best_f:
                        run_best_f = fy
                        run_best_x = y.copy()

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update mean
                new_x = sum(w[i] * offspring[i][1] for i in range(mu))
                new_x = np.clip(new_x, -5, 5)

                # Update step size control
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff/C)*inv_sqrtC@(new_x - x)
                diff = new_x - x
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrtC @ diff) / sigma

                # Update pc
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2/(dim+1)) * chi:
                    pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * diff / sigma

                # Update covariance
                delta_h = (1 + np.sign(np.linalg.norm(ps)/chi - 1) * (1 - np.sqrt(1 - cs)) - 1 + cs)
                C = (1 - c1 * (1 + (1 + cmu) * delta_h)) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    zi = inv_sqrtC @ (offspring[i][1] - x) / sigma
                    C += cmu * w[i] * np.outer(zi, zi)

                # Update inverse and sqrt
                C_inv = np.linalg.inv(C + 1e-12 * np.eye(dim))
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                sqrtC = B @ np.diag(D) @ B.T
                inv_sqrtC = B @ np.diag(1.0/D) @ B.T

                # Update sigma
                sigma *= np.exp(cs/ds * (np.linalg.norm(ps)/chi - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                x = new_x
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if fx < run_best_f:
                    run_best_f = fx
                    run_best_x = x.copy()

                # Check convergence
                if sigma < 1e-10 or np.max(D) / np.min(D) > 1e14:
                    break

                # If we're not making progress, consider restarting
                # (simplified: just continue)

            # If we still have budget and time, we could do more restarts
            if calls_used >= budget or time.time() > deadline - 10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
