"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random-sampling warm-start for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_clipped))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        # Use ~20% of budget for global exploration
        n_sample = min(80, budget // 5)
        for _ in range(n_sample):
            if time.time() > deadline - 30.0 or calls_made >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            # Fallback: try center
            x0 = np.zeros(dim)
            safe_call(x0)

        # Phase 2: CMA-ES from the best point
        if best_x is not None and calls_made < budget and time.time() < deadline - 10.0:
            self._cma_es(func, best_x.copy(), best_f, calls_made, safe_call, deadline, rng)

        # Phase 3: If budget remains, do a few more random perturbations around best
        if best_x is not None and calls_made < budget and time.time() < deadline - 2.0:
            sigma = 0.05
            for _ in range(50):
                if time.time() > deadline - 1.0 or calls_made >= budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-7:
                        break

        return best_x, best_f

    def _cma_es(self, func, x_mean, f_mean, calls_made, safe_call, deadline, rng):
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~9 for dim=5
        # Adjust pop_size to fit budget
        max_cma_calls = budget - calls_made - 10  # reserve some
        if max_cma_calls <= 0:
            return
        # Scale pop_size to fit
        max_generations = max(1, max_cma_calls // pop_size)
        if max_generations > 50:
            pop_size = max(4, max_cma_calls // 50)
            max_generations = max(1, max_cma_calls // pop_size)

        sigma = 0.5
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Adaptation rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[0] - 0.5) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for gen in range(max_generations):
            if time.time() > deadline - 10.0 or calls_made >= budget:
                break

            # Sample population
            offspring = []
            fvals = []
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline - 10.0:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky of C for sampling (approximate via C_sqrt_inv inverse)
                # For simplicity, use C directly: y = C @ z (this is an approximation)
                # Better: use eigendecomposition
                y = C @ z
                x_new = np.clip(x_mean + sigma * y, -5.0, 5.0)
                fx = safe_call(x_new)
                if fx is None:
                    break
                offspring.append(x_new)
                fvals.append(fx)

            if len(offspring) < mu:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            offspring = [offspring[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i]
            x_mean = x_new_mean

            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi_n * C_sqrt_inv @ (C @ (offspring[0] - x_mean) / sigma)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - gen * pop_size))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance
            v = (offspring[0] - x_mean) / sigma
            pc = (1 - cc) * pc + cc * v
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + pc)
            for i in range(mu):
                vi = (offspring[i] - x_mean) / sigma
                C += cmu * weights[i] * np.outer(vi, vi)

            # Re-orthogonalize C (ensure positive semi-definite)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_sqrt_inv approximation
            # Use inverse of C for the ps update (approximation)
            try:
                C_sqrt_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_sqrt_inv = np.eye(dim)

            # Track best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = offspring[0].copy()

            # Early stopping: if sigma is very small or no progress
            if sigma < 1e-8:
                break

        # After CMA-ES, update global best
        if best_x is not None:
            # Try the CMA-ES mean as final candidate
            if calls_made < budget and time.time() < deadline - 5.0:
                fx_mean = safe_call(x_mean)
                if fx_mean is not None and fx_mean < best_f:
                    best_f = fx_mean
                    best_x = x_mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
