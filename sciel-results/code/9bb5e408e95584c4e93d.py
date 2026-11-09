"""Rewritten optimizer using CMA-ES with adaptive restarts for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Run CMA-ES with a given budget."""
        n = self.dim
        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(n))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc - cs) / ((n + 1.3) ** 2 - 1)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 - 2))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Initialize state
        x = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Evaluate initial point
        x = np.clip(x, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        if budget <= 1:
            return best_x, best_f

        # Main loop
        while calls < budget:
            # Generate lambda_ offspring
            pop = np.zeros((lambda_, n))
            fvals = np.zeros(lambda_)
            
            for i in range(lambda_):
                # Sample from multivariate normal
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                pop[i] = y
                fvals[i] = func(y)
                calls += 1
                
                if calls >= budget:
                    break

            # If we hit budget, evaluate what we have
            if calls >= budget:
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = pop[min_idx].copy()
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Compute weighted mean of mu best points
            xnew = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            xnew = np.clip(xnew, -5, 5)

            # Update evolution paths
            ydiff = (xnew - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ydiff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) / chi_n) < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ydiff

            # Compute covariance matrix update
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_hs - c1) * np.eye(n)  # This is wrong, let me fix
            # Actually, C update: C = (1 - c1) * C + c1 * (outer + delta_hs * C_prev)
            # We need to keep C from previous iteration. Let me restructure.
            
            # Let me redo this properly with C stored
            # Actually, let me use a simpler approach: store C as a matrix
            # I'll restructure the CMA-ES to properly maintain C

            break  # Break out to fix the implementation

        return best_x, best_f

    def _cma_es_proper(self, func, x0, sigma0, budget, rng):
        """Proper CMA-ES implementation."""
        n = self.dim
        lambda_ = int(4 + 3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc - cs) / ((n + 1.3) ** 2 - 1)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 - 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)  # Covariance matrix
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls = 0

        x = np.clip(x, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        if budget <= 1:
            return best_x, best_f

        while calls < budget:
            # Generate population
            pop = np.zeros((lambda_, n))
            fvals = np.zeros(lambda_)
            
            # Pre-compute Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)

            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                pop[i] = y
                fvals[i] = func(y)
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = pop[min_idx].copy()
                break

            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            xnew = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            xnew = np.clip(xnew, -5, 5)

            ydiff = (xnew - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ydiff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) / chi_n) < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ydiff

            # Update C
            delta_hs = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hs - c1) * C + c1 * np.outer(pc, pc)
            C += cmu * np.sum(weights[i] * np.outer(pop[i] - x, pop[i] - x) / sigma ** 2 for i in range(mu))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            x = xnew

        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Multi-start CMA-ES with adaptive budget allocation
        # Start with a few initial points, then focus on the best
        num_starts = 5
        calls_per_start = max(1, (budget - num_starts) // (num_starts - 1)) if num_starts > 1 else budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Generate initial points
        init_points = []
        # One at origin
        init_points.append(np.zeros(n))
        # Random points
        for _ in range(num_starts - 1):
            init_points.append(rng.uniform(-4, 4, n))

        for i, x0 in enumerate(init_points):
            remaining = budget - calls_used
            if remaining <= 0:
                break

            # Allocate budget: first few get less, last one gets more
            if i < num_starts - 1:
                alloc = min(remaining, max(10, budget // (num_starts * 2)))
            else:
                alloc = remaining

            x0 = np.clip(x0, -5, 5)
            # Evaluate initial point
            f0 = func(x0)
            calls_used += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if calls_used >= budget:
                break

            # Run CMA-ES from this point
            x_best, f_best = self._cma_es_proper(func, x0, 2.0, alloc, rng)
            calls_used += alloc - 1  # -1 because we already evaluated x0

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        return best_x, best_f
# EVOLVE-BLOCK-END
