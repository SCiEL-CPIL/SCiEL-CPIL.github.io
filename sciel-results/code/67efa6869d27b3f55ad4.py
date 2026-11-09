"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a short deterministic Halton-based global exploration phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton(self, n, dim, base):
        """Generate n points in [0,1]^dim using a Halton-like sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            f = np.ones(n)
            b = float(base[d])
            for i in range(1, n):
                f[i] = 1.0 / b
                f[i - 1] = (f[i - 1] + 1.0) % 1.0
                f[i - 1] *= b
            # The standard Halton generation:
            # Let's use the proper algorithm
            result = np.zeros(n)
            for i in range(1, n):
                f = 1.0 / b
                d_val = i
                while d_val > 0:
                    d_val, rem = divmod(d_val, int(b))
                    f *= b
                    result[i] += rem * f
            points[:, d] = result
        return points

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # 20s safety margin before 300s kill

        # Phase 1: Deterministic global exploration with Halton sequence
        # Use ~30% of budget for global search
        n_global = max(10, int(budget * 0.3))
        n_global = min(n_global, budget - 20)  # leave at least 20 for local

        # Generate Halton points in [-5, 5]^dim
        # Use bases: first few primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        if dim <= len(primes):
            bases = primes[:dim]
        else:
            # Extend with more primes if needed
            bases = list(primes)
            p = 31
            while len(bases) < dim:
                bases.append(p)
                p += 2

        halton_pts = self._halton(n_global, dim, bases)
        # Map from [0,1]^dim to [-5,5]^dim
        halton_pts = halton_pts * 10.0 - 5.0

        # Evaluate all global points
        best_x = None
        best_f = np.inf
        evaluated = []

        for i in range(n_global):
            if time.time() > deadline:
                break
            x = halton_pts[i]
            fx = func(x)
            evaluated.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # If no evaluations were made (shouldn't happen), do one random
        if best_x is None:
            x = np.random.default_rng(42).uniform(-5, 5, dim)
            best_f = func(x)
            best_x = x.copy()
            evaluated.append((x, best_f))

        # Phase 2: CMA-ES local refinement starting from best global point
        # Use remaining budget
        remaining = budget - len(evaluated)
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        x_best = best_x.copy()
        f_best = best_f
        sigma = 1.0  # initial step size
        # CMA-ES state
        mean = x_best.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Precompute Cholesky
        C_sqrt = np.linalg.cholesky(C)

        # CMA-ES parameters
        mu = max(1, int(np.floor(dim / 2)))  # number of parents
        lambda_ = max(2 * mu, dim + 1)  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Track best
        best_x_es = best_x.copy()
        best_f_es = best_f

        calls_made = len(evaluated)
        es_calls = 0

        while calls_made + lambda_ <= budget and time.time() < deadline:
            # Generate population
            Y = []
            for k in range(lambda_):
                if time.time() > deadline:
                    break
                z = np.random.default_rng(hash((id(self), k, es_calls)) % (2**32)).standard_normal(dim)
                # Use C_sqrt to transform
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)
                calls_made += 1
                es_calls += 1

            if len(Y) < mu:
                break

            # Evaluate population
            F = []
            for y in Y:
                fy = func(y)
                F.append(fy)
                if fy < best_f_es:
                    best_f_es = fy
                    best_x_es = y.copy()

            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]

            # Update mean
            old_mean = mean.copy()
            mean = sum(w * Y[k] for k, w in enumerate(weights))
            mean = np.clip(mean, -5, 5)

            # Update step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv @ (mean - old_mean))
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (es_calls + 1))) < (1 + 2 / (dim + 1)) * chi_n else 0.0

            # Update evolution paths
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (mean - old_mean)

            # Update covariance matrix
            delta = h_sigma * (c_1 + c_mu)
            rank_one = np.outer(mean - old_mean, mean - old_mean)
            rank_mu = sum(weights[k] * np.outer(Y[k] - old_mean, Y[k] - old_mean) for k in range(mu))
            C = (1 + delta * c_1 - delta - c_mu * mu_eff / (1 + delta)) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * (1 - c_c) * C) + c_mu * rank_mu
            # Symmetrize
            C = (C + C.T) / 2

            # Regularize C if needed
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-8 * np.eye(dim)
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Track best
            if F[0] < best_f_es:
                best_f_es = F[0]
                best_x_es = Y[0].copy()

        # Phase 3: Final polish with Nelder-Mead if we have time
        if time.time() < deadline - 10.0 and calls_made < budget - 5:
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func,
                    best_x_es,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                if result.fun < best_f_es:
                    best_f_es = result.fun
                    best_x_es = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Return the best found
        if best_f_es < best_f:
            return best_x_es, best_f_es
        else:
            return best_x, best_f
# EVOLVE-BLOCK-END
