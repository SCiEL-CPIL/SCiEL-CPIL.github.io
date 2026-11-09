"""CMA-ES with multi-start for Rastrigin rotated f15."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES optimizer for Rastrigin rotated in 5D with multi-start."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Multi-start CMA-ES: run several short CMA-ES instances from different starts
        # and keep the best result found so far.
        
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # Initial diverse sampling to find good starting points
        n_starts = 8
        start_points = [best_x.copy()]
        
        # Add some diverse starting points
        for i in range(n_starts - 1):
            if time.time() > deadline or calls_used >= budget:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            fx = func(x)
            calls_used += 1
            start_points.append(x.copy())
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Run CMA-ES from each start point, allocating budget proportionally
        # Give more budget to starts that found better values
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return (best_x, best_f)

        # Sort starts by their function value
        start_fvals = []
        for sp in start_points:
            # We already evaluated these, so we know their values
            # But we need to track them. Let's re-evaluate or store.
            # Actually, we stored best_f but not individual start fvals.
            # Let's just use the best start and a few others.
            pass

        # Simpler approach: run CMA-ES from the best start with most of the budget,
        # and from a few other starts with less budget.
        
        # Determine which starts to use
        # Use top 3 starts by function value (we need to track these)
        # Since we didn't track individual start fvals, let's just use:
        # 1. The overall best point
        # 2. A couple of random diverse points
        
        # Actually, let's restructure: do a quick initial search, then CMA-ES from best.
        
        # Phase 1: Quick random search to find a good basin
        n_random = min(50, remaining_budget // 4)
        if n_random > 0:
            for i in range(n_random):
                if time.time() > deadline or calls_used >= budget:
                    break
                x = rng.uniform(-4.5, 4.5, dim)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: CMA-ES from the best point found
        remaining_budget = budget - calls_used
        if remaining_budget > 20:
            best_x, best_f, calls_used = self._cma_es(
                func, best_x, best_f, remaining_budget, dim, rng, deadline, calls_used
            )

        # Phase 3: If budget remains, try CMA-ES from a few other diverse points
        remaining_budget = budget - calls_used
        if remaining_budget > 20:
            # Pick 2-3 other diverse starting points
            n_extra_starts = 2
            for s in range(n_extra_starts):
                if time.time() > deadline or calls_used >= budget:
                    break
                x_start = rng.uniform(-4.0, 4.0, dim)
                fx_start = func(x_start)
                calls_used += 1
                
                if fx_start < best_f:
                    best_f = fx_start
                    best_x = x_start.copy()
                    continue  # Don't run CMA-ES if this is already the best
                
                # Only run CMA-ES if this start is competitive
                if fx_start < best_f * 1.5 + 10:  # Heuristic: within some range
                    alloc = min(remaining_budget // (n_extra_starts - s), 100)
                    if alloc > 10:
                        x_s, f_s, calls_used = self._cma_es(
                            func, x_start, fx_start, alloc, dim, rng, deadline, calls_used
                        )
                        if f_s < best_f:
                            best_f = f_s
                            best_x = x_s.copy()
                        remaining_budget = budget - calls_used

        # Final polish: fine local search
        if time.time() < deadline and calls_used < budget:
            fine_radius = 0.2
            for i in range(min(20, budget - calls_used)):
                if time.time() > deadline:
                    break
                x = best_x + rng.normal(0, fine_radius / dim**0.5, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    fine_radius *= 0.7

        return (best_x, best_f)

    def _cma_es(self, func, x0, f0, budget, dim, rng, deadline, calls_used):
        """Run CMA-ES from x0 with given budget."""
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        mean = x0.copy()
        sigma = 1.0  # Initial sigma
        best_x = x0.copy()
        best_f = f0

        # Initialize evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # Will be updated

        # Precompute Cholesky-related stuff
        def update_chol():
            nonlocal C, C_sqrt_inv
            try:
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
                return True
            except:
                # If C is not positive definite, reset
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                return True

        update_chol()

        # CMA-ES loop
        while calls_used < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lam):
                if time.time() > deadline or calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt_inv @ z)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                pop.append((x, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(pop) < mu:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            selected = pop[:mu]

            # Update mean
            new_mean = np.zeros(dim)
            for i, (x, fx) in enumerate(selected):
                new_mean += weights[i] * x
            mean = new_mean

            # Update sigma (log-step-size adaptation)
            # Compute actual step size
            if mu_eff > 0:
                # Approximate actual step size
                actual_sigma = np.sqrt(np.mean([
                    np.sum((x - mean)**2) for x, _ in selected
                ])) / np.sqrt(dim)
                # Update sigma
                sigma *= math.exp(c_sigma / d_sigma * (actual_sigma / sigma - 1))

            # Update evolution paths
            # p_sigma update
            # Need to compute the mean of selected in the transformed space
            # For simplicity, use a simplified update
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.mean([
                (x - mean) / sigma for x, _ in selected
            ], axis=0)

            # p_c update
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * np.mean([
                (x - mean) / sigma for x, _ in selected
            ], axis=0)

            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c + (1 - (1 - c_c)^2) * I) + c_mu * sum(weights_i * (x_i - mean)(x_i - mean)^T)
            diff = np.zeros((dim, dim))
            for i, (x, fx) in enumerate(selected):
                d = (x - mean) / sigma
                diff += weights[i] * np.outer(d, d)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(dim)) + c_mu * diff

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update Cholesky
            update_chol()

            # Check for convergence
            if sigma < 1e-10:
                break

        return (best_x, best_f, calls_used)
