"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with Halton initialization for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline - 5

        def remaining():
            return budget - calls_made

        def halton(index, base):
            """Generate Halton sequence value for given index and base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim):
            """Generate a point in [0,1]^dim using Halton sequence."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = []
            for d in range(dim):
                point.append(halton(index + 1, primes[d]))
            return np.array(point)

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Halton sequence for systematic space-filling search
        n_halton = min(100, budget // 5)
        for i in range(n_halton):
            if calls_made >= budget or not time_ok():
                break
            x = 10.0 * halton_point(i, dim) - 5.0
            evaluate(x)

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: Multi-restart CMA-ES
        def cma_es(x0, sigma0, max_generations):
            nonlocal best_x, best_f, calls_made

            x = x0.copy()
            sigma = sigma0
            lambda_ = 8
            mu = 4
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            c1 = 1 / (dim + 3)
            cmu = 2 / ((dim + 2) ** 2 + 5)
            cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
            cs = (dim + 1.7) / (dim + 9)
            damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            for gen in range(max_generations):
                if calls_made >= budget or not time_ok():
                    break

                # Generate lambda_ offspring
                offspring = []
                for i in range(lambda_):
                    if calls_made >= budget or not time_ok():
                        break
                    z = np.random.standard_normal(dim)
                    y = L @ z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = evaluate(xi)
                    offspring.append((fi, xi))

                if len(offspring) == 0:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Take top mu
                mu_actual = min(mu, len(offspring))
                selected = [offspring[i][1] for i in range(mu_actual)]

                # Compute mean of selected
                new_x = np.zeros(dim)
                for i in range(mu_actual):
                    new_x += weights[i] * selected[i]

                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50)

                # Update covariance
                v = (new_x - x) / sigma if sigma > 0 else np.zeros(dim)
                rank_one = c1 * (np.outer(v, v) + (1 - hsig) * cc * (2 - cc) * np.outer(pc, pc))
                rank_mu = cmu * sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) / max(sigma ** 2, 1e-20) for i in range(mu_actual))
                C = (1 - c1 - cmu) * C + rank_one + rank_mu

                # Update pc
                pc = (1 - cc) * pc + cc * hsig * v

                x = new_x

                # Ensure C is symmetric positive definite and update Cholesky
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))

                # If converged, break
                if sigma < 1e-7:
                    break

            return best_x.copy(), best_f

        # First CMA-ES run from best Halton point
        cma_es(best_x, 1.0, 200)

        # Multi-restart: try a few more CMA-ES runs from different starting points
        restarts = 3
        for r in range(restarts):
            if calls_made >= budget or not time_ok():
                break

            # Pick a random starting point, biased toward current best
            if np.random.random() < 0.3:
                x_start = best_x + np.random.normal(0, 1.0, dim)
            else:
                x_start = np.random.uniform(-5, 5, dim)
            x_start = np.clip(x_start, -5, 5)
            evaluate(x_start)

            cma_es(x_start, 1.5, 100)

        return best_x, best_f
# EVOLVE-BLOCK-END
