"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with adaptive covariance for rotated landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng, state):
        """Run CMA-ES from initial point x0 with step size sigma0."""
        n = self.dim
        calls_made = state['calls_made']
        best_x = state['best_x']
        best_f = state['best_f']

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))
        cc = (4 + n / (4 + n)) / ((n + 4) + (4 + n / (4 + n)))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 1))
        cmu = min(0.3, (2 - cc + 2 / n) / ((n + 2) + cc + 2 / n))
        damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs

        # State variables
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C))

        def update_inv_sqrt_C(C_new):
            try:
                return np.linalg.inv(np.linalg.cholesky(C_new))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_new)
                eigvals = np.maximum(eigvals, 1e-12)
                return np.linalg.inv(np.diag(np.sqrt(eigvals)) @ eigvecs.T)

        for gen in range(10000):
            if state['calls_made'] >= self.budget - 5:
                break
            if time.time() > deadline - 5:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if state['calls_made'] >= self.budget - 5:
                    break
                z = rng.standard_normal(n)
                y = mean + sigma * (inv_sqrt_C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                state['calls_made'] += 1
                if fy < state['best_f']:
                    state['best_f'] = fy
                    state['best_x'] = y.copy()
                offspring.append((fy, y))

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            lam = len(offspring)
            mu = lam // 2

            # Weights
            weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
            weights = weights[:mu]
            weights = weights / weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][1]
            mean = new_mean

            # Update covariance
            y_inv_sqrt = np.zeros((mu, n))
            for i in range(mu):
                y_inv_sqrt[i] = (offspring[i][1] - mean) / sigma

            # Update path covariance
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (inv_sqrt_C @ (new_mean - mean))
            norm_pc = np.linalg.norm(pc)
            hsig = 1 if norm_pc / np.sqrt(1 - (1 - cc) ** (2 * (state['calls_made'] - self.budget + 5) / pop_size)) < (1 + 2 / (n + 1)) * np.sqrt(np.trace(C)) / n else 0

            # Update path evolution strategy
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (new_mean - mean))

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + cmu * (delta_hsig - 1) + 2 * cmu) * C
            for i in range(mu):
                yv = y_inv_sqrt[i]
                C += cmu * weights[i] * (np.outer(yv, yv) - C)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            inv_sqrt_C = update_inv_sqrt_C(C)

            # Update step size (CSA)
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(-2 * np.log(np.exp(-1))) - 1))
            sigma = np.clip(sigma, 1e-8, 10)

            # Convergence check
            if np.all(np.abs(mean - x0) < 1e-6) and sigma < 1e-6:
                break

        return state['best_x'], state['best_f']

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        rng = np.random.default_rng(0)

        state = {
            'calls_made': 0,
            'best_x': None,
            'best_f': np.inf
        }

        def tracked_func(x):
            state['calls_made'] += 1
            fx = func(x)
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
            return fx

        # Phase 1: Random sampling to find promising starting points
        n_random = 30
        candidates = []
        for i in range(n_random):
            if state['calls_made'] >= self.budget - 50:
                break
            if time.time() > deadline - 30:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = tracked_func(x)
            candidates.append((fx, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from top candidates
        top_k = min(3, len(candidates))
        for i in range(top_k):
            if state['calls_made'] >= self.budget - 10:
                break
            if time.time() > deadline - 15:
                break

            x0 = candidates[i][1]
            # Initial sigma based on distance to other candidates
            sigma0 = 1.0
            try:
                self._cma_es(func, x0, sigma0, self.budget, deadline, rng, state)
            except Exception:
                continue

        # Final check
        if state['best_x'] is None:
            state['best_x'] = np.zeros(self.dim)
            state['best_f'] = func(state['best_x'])
            state['calls_made'] += 1

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
