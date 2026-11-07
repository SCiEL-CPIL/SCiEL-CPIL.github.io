"""Optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: Sobol global exploration, then CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Sobol quasi-random sequence generator ---
        def sobol_sequence(m, n, offset=0.5, scale=1.0):
            """Generate m Sobol points in [0,1]^n, scaled to [-offset, offset]*scale."""
            # Use scipy if available, otherwise fall back to Halton
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=n, scramble=True, seed=12345)
                sampler.fast_forward(m)
                points = sampler.random()
                # Map from [0,1] to [-offset*scale, offset*scale]
                points = (points - 0.5) * 2 * offset * scale
                return points
            except ImportError:
                # Fallback: Halton sequence
                def halton(index, base):
                    result = 0.0
                    f = 1.0
                    i = index
                    while i > 0:
                        f /= base
                        result += f * (i % base)
                        i //= base
                    return result

                points = np.zeros((m, n))
                for i in range(m):
                    for j in range(n):
                        points[i, j] = (halton(i + 1, j + 2) - 0.5) * 2 * offset * scale
                return points

        # --- CMA-ES (1+λ) for local refinement ---
        def cma_es(x0, sigma0, max_calls, mu=1, lambda_=4):
            nonlocal total_calls, best_x, best_f
            if max_calls < 20:
                return x0, best_f

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            cs = 1.0
            c1 = (mu_eff + 2) / ((n + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            mu_eff = mu
            # Step size control
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Generate initial population
            def generate_population():
                zs = np.random.standard_normal((lambda_, n))
                B = np.linalg.cholesky(C)
                return zs @ B.T * sigma + x_mean

            # Evaluate initial population
            pop = generate_population()
            fvals = []
            for i in range(lambda_):
                f = try_eval(pop[i])
                if f is None:
                    return x_mean, best_f
                fvals.append(f)

            for _ in range(max_calls // lambda_):
                if total_calls >= budget - 5 or time.time() > deadline - 5:
                    break

                fvals = np.array(fvals)
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                weights = np.zeros(lambda_)
                weights[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                x_new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Update evolution path
                pc = (1 - cs) * pc + cs / np.sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** 2) / chi_n <
                        (1.4 + 2 / (n + 1)) * (1 - cs))
                cs = (1 - cs) * cs + hsig * cs
                C = (1 - cs * (c1 + cmu)) * C + cmu * np.outer(pc, pc) + \
                    c1 * (1 - (1 - cs) ** 2) * np.outer(x_new_mean - x_mean, x_new_mean - x_mean) / sigma ** 2

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(pc) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                x_mean = x_new_mean

                # Generate new population
                pop = generate_population()
                fvals = []
                for i in range(lambda_):
                    f = try_eval(pop[i])
                    if f is None:
                        return x_mean, best_f
                    fvals.append(f)

            return x_mean, best_f

        # --- Phase 1: Global exploration with Sobol sequence ---
        n_global = min(60, budget // 8)
        # Generate Sobol points covering [-4, 4]^5
        sobol_pts = sobol_sequence(n_global, n, offset=4.0, scale=1.0)
        for i in range(n_global):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            try_eval(sobol_pts[i])

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Local refinement with CMA-ES from best point ---
        remaining = budget - total_calls
        if remaining > 50:
            # Start with a reasonable sigma based on the spread of good points
            sigma0 = 1.0
            x_opt, f_opt = cma_es(best_x.copy(), sigma0, remaining)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 3: Final polish with tiny perturbations ---
        if total_calls < budget and best_x is not None and time.time() < deadline - 3:
            n_final = min(20, budget - total_calls)
            rng = np.random.default_rng(999)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 2:
                    break
                scale = 0.05 * (1.0 - i / n_final) + 0.005
                x_try = best_x + rng.normal(0, scale, n)
                try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
