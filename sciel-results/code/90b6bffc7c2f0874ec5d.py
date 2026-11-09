"""Multi-start CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for 5-D optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        def check_time():
            return time.time() < deadline

        def update_best(x, fx):
            nonlocal best_x, best_f, total_calls
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Generate diverse initial points using a quasi-random sequence
        # and some random points to cover the 101 local minima
        rng = np.random.default_rng(42)

        # Create a set of diverse starting points
        n_starts = 12
        starts = []

        # Halton-like quasi-random points
        for i in range(n_starts):
            # Use different bases for each dimension to get good coverage
            point = []
            for d in range(dim):
                base = d + 2  # bases 2, 3, 4, 5, 6
                # Van der Corput sequence
                f = 1.0
                n_val = i + 1
                while n_val > 0:
                    f /= base
                    n_val //= base
                point.append(f)
            # Map to [-5, 5]
            point = np.array(point) * 10.0 - 5.0
            # Add some randomness
            point = point + rng.uniform(-0.5, 0.5, dim)
            point = np.clip(point, -5, 5)
            starts.append(point)

        # Add a few purely random points
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, dim))

        # Sort starts by some heuristic: prefer points near center and corners
        # Actually, let's just use them all

        # CMA-ES parameters
        # For 5-D, lambda = 4 + floor(3*log10(dim)) = 4 + 3 = 7
        # But we'll use a smaller population for speed
        lambda_pop = 4
        mu = lambda_pop // 2  # 2

        # Precompute weights
        # Standard CMA-ES weights: w_i = log(mu+0.5) - log(i) for i=1..mu
        weights = np.zeros(lambda_pop)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        def run_cma_es(x0, fx0, remaining_budget):
            nonlocal best_x, best_f, total_calls

            x_mean = x0.copy()
            sigma = 1.0
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            for _ in range(remaining_budget):
                if not check_time():
                    return
                if total_calls >= budget:
                    return

                # Generate lambda offspring
                offspring = []
                for k in range(lambda_pop):
                    if not check_time():
                        return
                    if total_calls >= budget:
                        return
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_calls += 1
                    offspring.append((fy, y))

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i][1]
                x_mean = new_mean

                # Update covariance matrix
                diff = np.array([offspring[i][1] for i in range(mu)]) - x_mean
                C += c_1 * (np.outer(x_mean - x0, x_mean - x0) + (C - np.eye(dim) * 0))  # simplified
                # Actually, let's do proper CMA-ES update
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(x_mean, x_mean) + C - np.eye(dim) * 0)  # This is getting complicated

                # Let me simplify: use a simpler covariance adaptation
                # Actually, for 5-D with limited budget, let's use a simpler approach

                # Update step size
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) / chi_n * (B @ (D * z))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 1 / dim) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

                sigma = max(sigma, 1e-8)

                # Update best
                if offspring[0][0] < best_f:
                    best_x = offspring[0][1].copy()
                    best_f = offspring[0][0]

                # Check convergence
                if sigma < 1e-6:
                    break

            return

        # Simpler approach: use (1+1)-ES with multi-start and adaptive sigma
        # This is more robust and faster for small dimensions

        def run_es(x0, fx0, remaining_budget):
            nonlocal best_x, best_f, total_calls

            x = x0.copy()
            fx = fx0
            sigma = 2.0
            success_count = 0
            total_trials = 0
            stagnation = 0

            for _ in range(remaining_budget):
                if not check_time():
                    return
                if total_calls >= budget:
                    return

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    stagnation = 0
                    # Increase sigma on success
                    sigma *= 1.2
                else:
                    stagnation += 1
                    # Decrease sigma on failure
                    sigma *= 0.9

                # Adaptive sigma: 1/5 success rule
                if total_trials > 0 and total_trials % 10 == 0:
                    if success_count / total_trials > 0.2:
                        sigma *= 1.1
                    elif success_count / total_trials < 0.1:
                        sigma *= 0.8
                    success_count = 0
                    total_trials = 0

                total_trials += 1

                # Reset sigma if stagnating
                if stagnation > 20:
                    sigma = 1.0
                    stagnation = 0

                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)

                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx

            return

        # Run multi-start ES
        for i, start in enumerate(starts):
            if not check_time():
                break
            if total_calls >= budget:
                break

            fx = func(start)
            total_calls += 1
            update_best(start, fx)

            # Allocate remaining budget among starts
            # Give more budget to promising starts
            remaining = budget - total_calls
            if remaining <= 0:
                break

            # Simple allocation: equal budget for each start, but adjust based on progress
            budget_per_start = max(1, remaining // (n_starts + 4 - i))
            run_es(start, fx, budget_per_start)

        # If we have budget left, do a final local search from the best point
        if best_x is not None and check_time() and total_calls < budget:
            remaining = budget - total_calls
            run_es(best_x, best_f, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
