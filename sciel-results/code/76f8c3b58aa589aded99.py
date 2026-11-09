"""Portfolio search optimizer for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


class Optimizer:
    """CMA-ES with restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return f

        def cmaes(x0, sigma0, max_iter, seed):
            """Run CMA-ES from x0 with initial step sigma0."""
            nonlocal best_x, best_f, calls_made, deadline

            rng = np.random.default_rng(seed)
            n = dim
            # CMA-ES parameters
            lambda_ = int(4 + 3 * np.log(n))  # population size
            mu = lambda_ // 2  # recombination weight count
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1)) + cs
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)
            x = np.array(x0, dtype=float)
            fbest = eval_func(x)
            best_local_x = x.copy()
            best_local_f = fbest

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate population
                pops = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ z) / D
                    pops.append(y)

                # Evaluate
                fitness = [(eval_func(p), i) for i, p in enumerate(pops)]
                fitness.sort(key=lambda t: t[0])

                # Update best
                if fitness[0][0] < best_local_f:
                    best_local_f = fitness[0][0]
                    best_local_x = pops[fitness[0][1]].copy()
                    if best_local_f < best_f:
                        best_f = best_local_f
                        best_x = best_local_x.copy()

                # Recombination
                xnew = sum(weights[i] * pops[fitness[i][1]] for i in range(mu))

                # Update path
                pc = (1 - cc) * pc + cc / np.sqrt(cc) * (B @ D) @ (xnew - x) / sigma
                norm_pc = np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (it + 1)) / np.sqrt(n)
                if norm_pc < (1.4 + 2 / (n + 1)) * (1 - cc):
                    C = B @ np.diag(1 + c1 * (cmu / mueff + c1 - 1)) @ B.T if False else B @ B.T  # simplified
                    # Actually let's do proper covariance update
                    # For simplicity and speed, use simplified version
                    pass

                # Update covariance matrix (simplified for speed)
                # Use rank-one and rank-mu updates
                diff = (xnew - x) / sigma
                C = B @ B.T  # This is O(n^3) but n=5 so it's fine
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (2 - cc) if norm_pc > (1.4 + 2 / (n + 1)) * (1 - cc) else 0) * C)
                for i in range(mu):
                    zi = B @ D @ (pops[fitness[i][1]] - x) / sigma
                    C += cmu * weights[i] * np.outer(zi, zi)

                # Eigen decomposition for B and D
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    D = 1.0 / np.sqrt(eigvals)
                    B = eigvecs
                except:
                    pass

                x = xnew

                # Step-size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.sqrt(cc) * B @ D @ diff
                hsig = (1 - 6 / (n + 1)) ** ((it + 1) * cs)
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1)) / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                if it % 5 == 0 and calls_made >= budget:
                    break

            return best_local_x, best_local_f

        # Strategy: Use multiple restarts with different initializations
        # First do a quick random search to find a good starting point
        rng_init = np.random.default_rng(42)
        x_init = rng_init.uniform(-5, 5, dim)
        f_init = eval_func(x_init)
        best_x = x_init.copy()
        best_f = f_init

        # Try several different starting points and step sizes
        seeds = [0, 1, 2, 3, 4]
        sigmas = [2.0, 1.0, 0.5]
        max_iter_per_run = 50

        for seed in seeds:
            if time.time() > deadline or calls_made >= budget:
                break
            for sigma0 in sigmas:
                if time.time() > deadline or calls_made >= budget:
                    break
                # Start from best found so far or a new random point
                if seed == 0:
                    x_start = x_init
                else:
                    x_start = best_x + 0.5 * rng_init.standard_normal(dim)
                    x_start = np.clip(x_start, -5, 5)
                _, _ = cmaes(x_start, sigma0, max_iter_per_run, seed * 100 + int(sigma0 * 10))

        # Final polish with a small-step local search
        if time.time() < deadline and calls_made < budget:
            sigma_final = 0.01
            for _ in range(min(20, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma_final * rng_init.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                else:
                    sigma_final *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
