"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """IPOP-CMA-ES with adaptive restarts for BBOB f18 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        n = self.dim
        best_x = None
        best_f = np.inf
        total_calls = 0
        rng = np.random.default_rng(42)

        # IPOP-CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        # For dim=5: lam = max(4, int(4+3*1.609)) = max(4, 8) = 8

        # Restart strategy: multiple restarts with different initializations
        max_restarts = 8
        sigma_init = 2.0
        x0_init = np.zeros(n)

        for restart in range(max_restarts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break

            # For each restart, use a different starting point
            if restart == 0:
                x0 = rng.uniform(-3, 3, n)
            else:
                # Perturb around previous best or use random
                if best_x is not None and restart < max_restarts // 2:
                    x0 = best_x + rng.normal(0, 1.0, n)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, n)

            sigma = sigma_init
            C = np.eye(n)  # covariance matrix
            B = np.eye(n)  # eigenvectors
            D = np.ones(n)  # eigenvalues

            # CMA-ES state
            ps = np.zeros(n)  # evolution path for sigma
            pc = np.zeros(n)  # evolution path for C
            cs = 1 / (np.sqrt(n) + 2)  # step size for sigma
            c1 = 2 / ((n + 1.3) ** 2 + lam)  # weight for rank-one update
            cmu = min(1 - c1, 2 * (n - 2 + 1 / n) / ((n + 2) ** 2 + lam))  # weight for rank-mu update
            mu = lam // 2  # number of best solutions used for update
            weights = np.log(np.arange(lam, 0, -1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights[:mu] ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)  # step size for C
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initialize
            x = x0.copy()
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check if we should stop this restart
            max_iter = max(1, (self.budget - total_calls) // lam)
            if restart > 0:
                # Limit iterations for later restarts
                max_iter = min(max_iter, max(5, 50 - restart * 5))

            for iteration in range(max_iter):
                if total_calls >= self.budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                # Generate population
                ys = []
                for i in range(lam):
                    bi = B @ (D * rng.standard_normal(n))
                    yi = np.clip(x + sigma * bi, -5, 5)
                    yi = np.asarray(yi, dtype=float)
                    fi = func(yi)
                    total_calls += 1
                    ys.append((fi, yi))
                    if fi < best_f:
                        best_f = fi
                        best_x = yi.copy()

                # Sort by fitness
                ys.sort(key=lambda t: t[0])

                # Update mean
                new_x = np.zeros(n)
                for i in range(mu):
                    new_x += weights[i] * ys[i][1]
                x = new_x

                # Update evolution paths
                dmean = (x - x0) / sigma if sigma > 1e-20 else np.zeros(n)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * dmean
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * dmean

                # Update covariance matrix
                delta_c = (1 - cc) * (2 * cc - 1) if hsig else 0
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (ys[i][1] - x0) / sigma if sigma > 1e-20 else np.zeros(n)
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta_c - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Eigendecomposition (less frequent to save time)
                if iteration % 5 == 0 or iteration == 0:
                    try:
                        D2, B2 = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D2, 1e-20))
                        B = B2
                    except:
                        pass

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # IPOP: check for stagnation and restart
                if iteration > 10:
                    # Check if progress is slow
                    f_best_now = ys[0][0]
                    f_prev_best = ys[mu][0] if mu < lam else f_best_now
                    if f_best_now - f_prev_best < 1e-10:
                        # Stagnation detected
                        sigma *= 0.5
                        x = best_x + rng.normal(0, 0.5, n)
                        x = np.clip(x, -5, 5)
                        C = np.eye(n)
                        B = np.eye(n)
                        D = np.ones(n)
                        ps = np.zeros(n)
                        pc = np.zeros(n)
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
