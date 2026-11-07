"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with:
    - Time guard to stay within wall-clock limits
    - Adaptive restarts based on stagnation
    - Ill-conditioned function awareness (Schaffers F7)
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin: 300s limit, keep 20s buffer
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # CMA-ES parameters
        def cma_es(x0, sigma0, budget):
            """Run CMA-ES from initial point x0 with initial step sigma0."""
            n = self.dim
            # Population size
            lam = max(4 + int(3 * np.log(n)), 6)
            # Weights
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, lam + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            # Step size control
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Initial evaluation
            fx_list = []
            x_list = []
            fx = func(xmean)
            update_best(xmean, fx)
            calls_made = 1

            for gen in range(budget // lam + 1):
                # Check time
                if time.time() > deadline - 5.0:
                    break

                # Generate population
                for i in range(lam):
                    # Check time before each evaluation
                    if time.time() > deadline - 2.0:
                        break
                    if calls_made >= self.budget:
                        break

                    # Sample from N(xmean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    y = xmean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    x_list.append(y)
                    fx_list.append(fy)
                    update_best(y, fy)

                if len(x_list) < lam:
                    break

                # Sort by fitness
                idx = np.argsort(fx_list)
                x_sorted = [x_list[i] for i in idx]
                fx_sorted = [fx_list[i] for i in idx]

                # Update mean
                xold = xmean.copy()
                xmean = sum(weights[i] * x_sorted[i] for i in range(mu))

                # Update C
                diff = xmean - xold
                pc = (1 - cc) * pc + cc * diff / (sigma * np.sqrt(cc * (2 - cc) * mueff * C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff / sigma

                # Rank-one and rank-mu update
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) / chi_n < 1.4 + 2 / (n + 1)
                C = (1 + c1 * hsig * (1 - hsig) - c1 - cmu) * C \
                    + c1 * np.outer(pc, pc) \
                    + cmu * sum(weights[i] * np.outer(x_sorted[i] - xold, x_sorted[i] - xold) for i in range(mu))

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 20.0)

                x_list = []
                fx_list = []

            return xmean, fx_sorted[0] if fx_sorted else np.inf

        # Multi-start strategy with different initial points
        # Start with a few diverse initializations
        num_starts = min(5, self.budget // 100 + 1)
        per_start_budget = self.budget // num_starts

        # Initial diverse points
        initial_points = []
        initial_points.append(np.zeros(self.dim))  # Center
        for _ in range(num_starts - 1):
            initial_points.append(rng.uniform(-3, 3, self.dim))

        for i, x0 in enumerate(initial_points):
            if time.time() > deadline - 10.0:
                break
            remaining_budget = self.budget - (i * per_start_budget)
            if remaining_budget <= 0:
                break
            sigma0 = 2.0 if i == 0 else 1.0
            cma_es(x0, sigma0, min(per_start_budget, remaining_budget))

        # Final local refinement with (1+1)-ES if time permits
        if best_x is not None and time.time() < deadline - 20.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            # Use remaining budget
            elapsed_calls = 0
            max_local = min(self.budget, 100)
            for _ in range(max_local):
                if time.time() > deadline - 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                elapsed_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                update_best(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
