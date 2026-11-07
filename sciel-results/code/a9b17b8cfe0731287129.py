"""Hybrid global-exploration + CMA-ES optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global LHS exploration followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() > deadline - 2.0:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                return None
            calls_used += 1
            return x.copy(), fx

        # Phase 1: Global exploration with Latin Hypercube Sampling
        n_global = min(60, budget)
        candidates = []
        for i in range(n_global):
            # LHS: divide each dimension into n_global strata, pick one point per stratum
            strata = np.linspace(-5, 5, n_global + 1)
            x = np.zeros(dim)
            for d in range(dim):
                lo = strata[i % n_global]
                hi = strata[(i + 1) % n_global]
                x[d] = rng.uniform(lo, hi)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        # Also add some uniform random points for coverage
        n_extra = min(20, budget - calls_used)
        for _ in range(n_extra):
            x = rng.uniform(-5, 5, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        if not candidates:
            return np.zeros(dim), np.inf

        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]

        # Phase 2: CMA-ES local refinement
        remaining = budget - calls_used
        if remaining < 10:
            return best_x, best_fx

        # Initialize CMA-ES around the best point
        x_mean = best_x.copy()
        sigma = 1.0  # initial step size

        # CMA-ES parameters
        lambda_pop = max(8, int(4 + 3 * np.log(dim)))
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
        cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        C = np.eye(dim)
        csqrt = np.eye(dim)

        def update_sqrt(C_new):
            try:
                w, V = np.linalg.eigh(C_new)
                w = np.maximum(w, 1e-14)
                return V @ np.diag(np.sqrt(w)) @ V.T
            except Exception:
                return np.eye(dim)

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        no_improve_generations = 0
        best_fx_cma = best_fx
        best_x_cma = best_x.copy()

        while calls_used < budget and time.time() < deadline - 3.0:
            # Generate population
            pop = []
            for i in range(lambda_pop):
                if calls_used >= budget or time.time() > deadline - 2.0:
                    break
                try:
                    zi = rng.standard_normal(dim)
                    xi = x_mean + sigma * (csqrt @ zi)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    fi = float(fi)
                    calls_used += 1
                    pop.append((fi, xi))
                except Exception:
                    break

            if len(pop) == 0:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

            # Update best
            if pop[0][0] < best_fx_cma:
                best_fx_cma = pop[0][0]
                best_x_cma = pop[0][1].copy()
                no_improve_generations = 0
            else:
                no_improve_generations += 1

            # Check for stagnation: restart with smaller sigma
            if no_improve_generations > 20 and calls_used < budget - 30:
                x_mean = best_x_cma + rng.normal(0, 0.2, dim)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = max(0.1, sigma * 0.5)
                ps = np.zeros(dim)
                pc = np.zeros((dim, dim))
                C = np.eye(dim)
                csqrt = np.eye(dim)
                no_improve_generations = 0
                continue

            # Update mean
            n_use = min(mu, len(pop))
            x_new = np.zeros(dim)
            for i in range(n_use):
                x_new += weights[i] * pop[i][1]

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x_mean))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - n_global - n_extra + 1)))) < \
                   (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x_mean)

            # Update covariance matrix
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(n_use):
                y = (pop[i][1] - x_mean) / sigma
                rankmu += weights[i] * np.outer(y, y)

            C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu

            # Ensure C is symmetric and PSD
            C = (C + C.T) / 2
            try:
                w, V = np.linalg.eigh(C)
                w = np.maximum(w, 1e-14)
                C = V @ np.diag(w) @ V.T
                csqrt = update_sqrt(C)
            except Exception:
                csqrt = np.eye(dim)
                C = np.eye(dim)

            # Update step size
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 0.001, 15.0)

            x_mean = x_new

        # Phase 3: Final local refinement with smaller steps
        remaining = budget - calls_used
        if remaining > 3 and time.time() < deadline - 1.0:
            local_sigma = max(0.02, sigma * 0.3)
            for _ in range(min(remaining, 20)):
                if time.time() > deadline - 0.5:
                    break
                try:
                    y = best_x_cma + local_sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = float(func(y))
                    calls_used += 1
                    if fy < best_fx_cma:
                        best_fx_cma = fy
                        best_x_cma = y.copy()
                        local_sigma *= 1.1
                    else:
                        local_sigma *= 0.85
                except Exception:
                    break

        return best_x_cma, best_fx_cma
# EVOLVE-BLOCK-END
