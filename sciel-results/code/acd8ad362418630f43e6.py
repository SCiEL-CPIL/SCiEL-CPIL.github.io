"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and Lunacek basin-mirroring prior."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget_remaining, deadline, calls_used):
        """Run CMA-ES from x0 with initial sigma. Returns (best_x, best_fx, calls_used)."""
        dim = self.dim
        n = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(n))  # standard CMA-ES lambda
        mu = lambda_ // 2  # number of parents
        # Weights
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size control
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
        # Covariance matrix adaptation
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = (2 - cc + 2 * mueff / (n - 1)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Initial values
        xmean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        # Cholesky of C
        C_eigvals = np.ones(n)
        # Track best
        best_x = x0.copy()
        best_fx = np.inf
        calls = 0

        # Max iterations to prevent infinite loop
        max_iter = 200

        for it in range(max_iter):
            if calls >= budget_remaining or time.time() > deadline:
                break

            # Generate population
            for i in range(lambda_):
                if calls >= budget_remaining or time.time() > deadline:
                    break
                z = np.random.standard_normal(n)
                # Use Csqrt (approximate) for sampling
                y = Csqrt @ z
                x_new = xmean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()

            if calls >= budget_remaining or time.time() > deadline:
                break

            # Sort by fitness
            # We need to re-evaluate? No, we stored them. But we didn't store all.
            # Let's restructure: store all offspring and fitnesses.
            # Actually, let's redo the loop to store offspring.
            pass

        return best_x, best_fx, calls

    def _cma_es_stored(self, func, x0, sigma0, budget_remaining, deadline, calls_used, rng):
        """CMA-ES that stores offspring for proper sorting."""
        n = self.dim
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = (2 - cc + 2 * mueff / (n - 1)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        xmean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        best_x = x0.copy()
        best_fx = np.inf
        calls = calls_used

        # Initial evaluation
        if calls < budget_remaining:
            best_fx = func(x0)
            calls += 1
            if time.time() > deadline:
                return best_x, best_fx, calls

        max_iter = 150

        for it in range(max_iter):
            if calls >= budget_remaining or time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            fitnesses = []
            for i in range(lambda_):
                if calls >= budget_remaining or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                # Use Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                y = L @ z
                x_new = xmean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1
                offspring.append(x_new)
                fitnesses.append(fx_new)
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()

            if len(offspring) < 2 or calls >= budget_remaining or time.time() > deadline:
                break

            # Sort offspring
            idx = np.argsort(fitnesses)
            offspring = np.array(offspring)
            fitnesses = np.array(fitnesses)
            offspring = offspring[idx]
            fitnesses = fitnesses[idx]

            # Update mean
            xmean_old = xmean.copy()
            xmean_new = np.sum(weights[:, None] * offspring[:mu], axis=0)
            xmean = xmean_new

            # Update step size
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            ymean = np.sum(weights[:, None] * offspring[:mu], axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ ((xmean - xmean_old) / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(n):
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(n) - 1))

            # Update evolution path
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (L.T @ ((xmean - xmean_old) / sigma))

            # Update covariance
            delta_sig = (1 + 2 * max(0, (np.linalg.norm(ps) / np.sqrt(n) - 1) / ds)) * sigma - sigma
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None, None] * np.outer(offspring[:mu] - xmean_old, offspring[:mu] - xmean_old), axis=0)
            C = (1 + c1 + cmu - cc) * C + c1 * rank_one + cmu * rank_mu
            # Regularize
            C = (C + C.T) / 2

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Stagnation check: if sigma is very small, break
            if sigma < 1e-6:
                break

            # If no improvement for many iterations, break
            if it > 0 and fitnesses[0] >= best_fx + 1e-10:
                pass  # Continue, but we'll rely on sigma collapse or external restart

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        global_x = None
        global_fx = np.inf
        calls_used = 0

        # Lunacek offset: the local minimum is at a known offset from the global.
        # For BBOB f24, the offset is approximately (2.5, 2.5, ...) in the rotated space.
        # We use a heuristic: mirror the current best through the origin or a fixed offset.
        # The known offset for Lunacek is such that the local min is at x_global + offset.
        # A common offset is (0, 0, ..., 0) + (2.5 * sign) but it's instance-dependent.
        # We'll use a simple heuristic: try points that are "far" from the current best.

        max_restarts = 15
        # Budget allocation: give more budget to early restarts
        # First restart gets 40% of budget, subsequent ones get less
        budget_alloc = []
        remaining = budget
        for r in range(max_restarts):
            if r == 0:
                alloc = int(remaining * 0.35)
            elif r == 1:
                alloc = int(remaining * 0.25)
            elif r == 2:
                alloc = int(remaining * 0.20)
            else:
                alloc = int(remaining * 0.10)
            alloc = max(alloc, 20)  # minimum 20 evals per restart
            if alloc >= remaining:
                alloc = remaining
            budget_alloc.append(alloc)
            remaining -= alloc
            if remaining <= 0:
                break

        for restart, alloc in enumerate(budget_alloc):
            if calls_used >= budget or time.time() > deadline:
                break

            # Choose starting point
            if restart == 0:
                x0 = rng.uniform(-4, 4, dim)
            else:
                if global_x is not None:
                    # Mirror the global best through a random point to explore the other basin
                    # The Lunacek local minimum is at a fixed offset from the global.
                    # We try to jump to a distant region.
                    if rng.random() < 0.5:
                        # Jump to a random distant point
                        x0 = rng.uniform(-5, 5, dim)
                    else:
                        # Perturb the global best with large noise
                        noise = 3.0 * rng.standard_normal(dim)
                        x0 = np.clip(global_x + noise, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

            sigma0 = 1.5 if restart == 0 else 2.0

            # Run CMA-ES
            best_x, best_fx, calls_used = self._cma_es_stored(
                func, x0, sigma0, budget - calls_used, deadline, calls_used, rng
            )

            if best_fx < global_fx:
                global_fx = best_fx
                global_x = best_x.copy()

            # If we've found a very good solution, we can stop
            # (Heuristic: if best_fx is very low, assume we found the global min)
            # But we don't know f_opt, so we just continue if budget remains.

        return global_x, global_fx
# EVOLVE-BLOCK-END
