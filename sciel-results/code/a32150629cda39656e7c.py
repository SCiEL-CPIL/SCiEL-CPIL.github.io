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
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = (2 - mueff / D + cc) / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs

        x_mean = np.zeros(D)
        sigma = 4.0

        pc = np.zeros(D)
        ps = np.zeros(D)
        C = np.eye(D)

        rng = np.random.RandomState(42)

        def sample_from_C(C, sigma, rng):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            z = rng.standard_normal(D)
            return sigma * (eigvecs @ (np.sqrt(eigvals) * z))

        try:
            while call_count < budget and time.time() < deadline:
                # Generate population
                pop_y = []
                pop_x = []
                for i in range(lam):
                    y = sample_from_C(C, sigma, rng)
                    x = x_mean + y
                    pop_y.append(y)
                    pop_x.append(x)

                # Evaluate
                fitness = [eval_func(x) for x in pop_x]

                # Sort
                idx = np.argsort(fitness)
                y_sorted = [pop_y[i] for i in idx]
                x_sorted = [pop_x[i] for i in idx]

                # Update mean
                x_mean_new = sum(weights[i] * x_sorted[i] for i in range(mu))

                # Update sigma (CSA)
                ps = (1 - cs) * ps + cs * np.sqrt(cs / (1 - cs)) * (x_mean - x_mean_new) / sigma
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (D + 1)) * np.sqrt(D + 1) else 0
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(D) - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update C
                pc = (1 - cc) * pc + cc * hsig * (x_mean - x_mean_new) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 + 2 * cs) * cc) * C)
                for i in range(mu):
                    C += (cmu * weights[i]) * np.outer(y_sorted[i], y_sorted[i])
                C = 0.5 * (C + C.T)

                x_mean = x_mean_new

                if sigma < 1e-8:
                    break

        except (TimeoutError, ValueError):
            pass
        except Exception:
            pass

        return (best_x, best_f)
# EVOLVE-BLOCK-END
