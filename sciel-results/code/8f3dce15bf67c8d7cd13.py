"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Rosenbrock-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        calls_used = 0
        max_calls = self.budget

        def time_ok():
            return time.time() < deadline

        def call_count_ok():
            return calls_used < max_calls

        def wrapped(x):
            nonlocal calls_used
            if not call_count_ok():
                return float('inf')
            calls_used += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            return val

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_remaining, rng):
            nonlocal calls_used, best_x, best_f
            d = self.dim
            n = max(1, int(np.ceil(4 + 3 * np.log(d))))  # population size
            # Ensure we don't exceed remaining budget
            n = min(n, max(1, budget_remaining // 2))

            # Initialize
            x = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(d)
            D = np.ones(d)
            C = np.eye(d)
            ps = np.zeros(d)
            hc = 1.0 + 2.0 / max(1, d - 2) - 1.0
            cs = (np.sum(np.sqrt(D)) / np.sum(D)) * (1 + 2.0 / max(1, d - 2) - 1.0)
            cmu = min(0.2, 2.0 / ((d + 1.0) ** 2 - 2))
            cc = (4.0 + cs / cs - 1.0) / ((d + 2.0) * (1.0 - cs / cs))
            chiN = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

            # Initial evaluation
            fx = wrapped(x)
            update_best(x, fx)
            if not time_ok() or not call_count_ok():
                return

            # Main loop
            while time_ok() and call_count_ok():
                # Generate population
                Y = rng.standard_normal((n, d))
                X = np.array([x + sigma * (Y[i] @ B) * D for i in range(n)])
                F = np.array([wrapped(xi) for xi in X])

                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]

                # Update best
                if F[0] < best_f:
                    update_best(X[0], F[0])

                # Weighted mean
                w = np.log(np.arange(1, n + 1) - np.log(np.arange(1, n + 1) * (n - 1) / n))
                w = np.maximum(w, 0)
                w = w / np.sum(w)
                mu_eff = 1.0 / np.sum(w[:n // 2 + 1] ** 2)

                # Update mean
                x_new = np.sum([w[i] * X[i] for i in range(n)], axis=0)

                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chiN * (B @ D) @ ((x_new - x) / sigma)

                # Update C
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // n + 1)))) < (1.4 + 2 / (d + 1)) * chiN
                C = (1 - cmu * mu_eff - cc * cc + (cc * (2 - cc) if hsig else 0)) * C
                C += cmu * np.sum([w[i] * np.outer(D @ (B.T @ X[i]), D @ (B.T @ X[i])) for i in range(n)], axis=0)
                if hsig:
                    C += cc * np.outer(ps, ps)

                # Update B and D
                D2, B2 = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-20))
                B = B2

                # Update sigma
                sigma *= np.exp(cs / (5 * d) * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                x = x_new

                # Boundary handling: reflect if out of bounds
                for i in range(d):
                    if x[i] < -5:
                        x[i] = -5 + (-5 - x[i]) * 0.5
                    elif x[i] > 5:
                        x[i] = 5 - (x[i] - 5) * 0.5

                # Check for convergence
                if np.linalg.norm(x_new - x) < 1e-12 * max(1, np.linalg.norm(x)):
                    break

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # Start 1: Center
        x0 = np.zeros(self.dim)
        cma_es(x0, 2.0, max_calls, rng)

        if not time_ok() or not call_count_ok():
            return best_x, best_f

        # Start 2: Random point
        x0 = rng.uniform(-4, 4, self.dim)
        cma_es(x0, 1.5, max_calls - calls_used, rng)

        if not time_ok() or not call_count_ok():
            return best_x, best_f

        # Start 3: Another random point
        x0 = rng.uniform(-3, 3, self.dim)
        cma_es(x0, 1.0, max_calls - calls_used, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
