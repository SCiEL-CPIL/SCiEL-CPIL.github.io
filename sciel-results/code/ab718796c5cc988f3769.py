import numpy as np
import time

class Optimizer:
    """
    Hybrid Global-Local Optimizer for BBOB f14.
    Phase 1: CMA-ES (global exploration)
    Phase 2: (1+1)-ES (local exploitation/polishing)
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0  # 330s limit - 5s safety

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Bounds
        lb = -5.0
        ub = 5.0

        # Track best
        best_x = None
        best_f = float('inf')
        evals = 0

        def evaluate(x):
            nonlocal evals, best_x, best_f
            x = np.clip(x, lb, ub)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: CMA-ES (Global) ---
        # Allocate ~70% of budget to global search
        phase1_budget = int(budget * 0.7)

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # lambda = 4 + 3*ln(dim)
        pop_size = max(pop_size, 4)

        # Initialize population
        pop = []
        pop_f = []

        # Initial random points
        for i in range(pop_size):
            x = rng.uniform(lb, ub, dim)
            fx = evaluate(x)
            pop.append(x)
            pop_f.append(fx)

        # Sort by fitness
        idx = np.argsort(pop_f)
        pop = [pop[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]

        # CMA-ES state
        mean = np.mean(pop[:pop_size//2], axis=0) if pop_size >= 2 else pop[0].copy()
        sigma = 0.5
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        mu = pop_size // 2  # number of parents
        weights = np.zeros(mu)
        w_sum = 0
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum += weights[i]
        weights /= w_sum
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((dim + 2)**2 + mu_eff))

        generation = 0
        while evals < phase1_budget:
            # Check time
            if time.time() > self.deadline:
                break

            # Generate offspring
            new_pop = []
            new_pop_f = []

            for i in range(pop_size):
                if evals >= phase1_budget:
                    break
                if time.time() > self.deadline:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Transform using Cholesky decomposition of C
                L = np.linalg.cholesky(C)
                y = mean + sigma * L @ z
                y = np.clip(y, lb, ub)
                fy = evaluate(y)
                new_pop.append(y)
                new_pop_f.append(fy)

            # Sort by fitness
            idx = np.argsort(new_pop_f)
            new_pop = [new_pop[i] for i in idx]
            new_pop_f = [new_pop_f[i] for i in idx]

            # Update CMA-ES state
            # Weighted mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (new_pop[i] - mean) / sigma

            # Update sigma
            norm_y_w = np.linalg.norm(y_w)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * d_sigma * (norm_y_w / chi_n - 1) * y_w / norm_y_w if norm_y_w > 1e-16 else p_sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_y_w / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 5.0)

            # Update correlation matrix C
            # Update pc
            p_c = (1 - c_c) * p_c + c_c * y_w
            delta_heav = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_c
            C = (1 + c_1 + c_mu) * C
            C -= c_1 * (np.outer(p_c, p_c) + (2 * c_c - c_c**2) * C)
            for i in range(mu):
                C -= c_mu * weights[i] * np.outer(new_pop[i] - mean, new_pop[i] - mean) / sigma**2

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(dim)

            # Update mean
            mean = np.mean(new_pop[:mu], axis=0) if mu > 0 else new_pop[0]
            mean = np.clip(mean, lb, ub)

            generation += 1

        # --- Phase 2: Local (1+1)-ES (Exploitation) ---
        # Use remaining budget
        remaining_budget = budget - evals

        if remaining_budget > 0:
            # Start from global best
            x = best_x.copy()
            fx = best_f
            sigma = 0.3  # Initial small step
            sigma_min = 1e-8
            sigma_max = 5.0

            # (1+1)-ES with success rule
            for _ in range(remaining_budget):
                if time.time() > self.deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, lb, ub)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    # Increase sigma
                    sigma *= 1.1
                    if sigma > sigma_max:
                        sigma = sigma_max
                else:
                    # Decrease sigma
                    sigma *= 0.9
                    if sigma < sigma_min:
                        sigma = sigma_min

        return best_x, best_f
