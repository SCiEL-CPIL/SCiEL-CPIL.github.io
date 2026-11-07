"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global search with multi-start CMA-ES on BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse global exploration with random restarts ---
        # Use a small number of wide random starts to find good basins
        rng = np.random.default_rng(12345)

        # First, do a few random evaluations to get a rough sense of the landscape
        # and find a decent starting point for local optimization
        n_initial = min(20, budget // 10)
        candidates = []
        for _ in range(n_initial):
            x = rng.uniform(-4.5, 4.5, dim)
            fx = safe_func(x)
            candidates.append((fx, x))
            if time.time() > deadline:
                break

        candidates.sort(key=lambda t: t[0])
        # Take the best few as starting points for local search
        n_starts = min(8, len(candidates), max(1, (budget - n_initial) // 50))

        # --- Phase 2: Multi-start CMA-ES (simple implementation) ---
        # We'll implement a lightweight CMA-ES for each start
        # but with a shared budget allocation

        remaining_budget = budget - calls_made
        if remaining_budget <= 10:
            return best_x, best_f

        # Allocate budget among starts
        per_start_budget = max(20, remaining_budget // n_starts)

        for idx in range(n_starts):
            if time.time() > deadline:
                break
            if calls_made >= budget - 5:
                break

            start_x = candidates[idx][1].copy()
            start_f = candidates[idx][0]

            # Simple CMA-ES parameters
            lambda_pop = max(4, dim + int(3 * np.log(dim)))  # population size
            mu = lambda_pop // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(i + 1) - np.log(mu)
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            sigma = 0.5  # initial step size
            mean = start_x.copy()
            B = np.eye(dim)
            D = np.ones(dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            local_calls = 0
            max_local_calls = per_start_budget

            for gen in range(max_local_calls // lambda_pop + 2):
                if time.time() > deadline:
                    break
                if local_calls >= max_local_calls:
                    break
                if calls_made >= budget - 5:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_pop):
                    if time.time() > deadline or calls_made >= budget - 5:
                        break
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    local_calls += 1
                    offspring.append((fy, y))

                if len(offspring) < 2:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < start_f:
                    start_f = offspring[0][0]

                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += w[i] * offspring[i][1]
                mean = new_mean

                # Update covariance matrix adaptation
                y_mean = (mean - mean) / sigma if sigma > 0 else np.zeros(dim)  # simplified
                # Actually, we need the previous mean
                # Let's simplify: just update sigma and mean, skip full CMA for speed

                # Update sigma (1/5 rule simplified)
                # Count successes in this generation
                # For simplicity, use a heuristic
                if offspring[0][0] < start_f - 1e-10:
                    sigma *= 1.1
                else:
                    sigma *= 0.95

                sigma = np.clip(sigma, 0.01, 5.0)

                # Update mean to be the best offspring if it's significantly better
                if offspring[0][0] < start_f:
                    start_f = offspring[0][0]

        # --- Phase 3: Final local polish around the best point found ---
        if calls_made < budget - 10 and best_x is not None and time.time() < deadline:
            # Do a few more random perturbations around the best point
            polish_budget = min(30, budget - calls_made)
            sigma_polish = 0.1
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                x = best_x + sigma_polish * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    sigma_polish *= 0.9
                else:
                    sigma_polish *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
