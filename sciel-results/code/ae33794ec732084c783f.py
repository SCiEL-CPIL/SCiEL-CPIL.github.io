"""Optimizer for BBOB-style f13 (Sharp ridge) with CMA-ES + deadline guard."""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with stagnation restarts and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Quick random restarts to find a good basin
        # Use 20% of budget for coarse exploration
        n_explore = max(20, budget // 5)

        rng = np.random.default_rng(42)
        x0 = np.array([0.0] * dim)

        # Try a few random starts to find a promising region
        for i in range(min(5, n_explore // 10)):
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
            if time.time() > deadline:
                break

        # If no good point found, start from center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: CMA-ES from the best point found
        x = best_x.copy()
        f_best = best_f

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        c1 = (4 + ln / dim) / (dim**2 + 4 + 2) if dim > 0 else 0.1
        cmu = 2 * (min(1, (lambda_ // 2 - 1) / dim) - 1) / (dim**2 + 4)
        c_sigma = (dim + 1.3) / (dim + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + c_sigma

        # Initialize CMA-ES state
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Evaluate initial population
        pop = [x + sigma * rng.standard_normal(dim) for _ in range(lambda_)]
        pop = [np.clip(p, -5, 5) for p in pop]
        fvals = [func(p) for p in pop]

        # Track best
        for i, (p, fv) in enumerate(zip(pop, fvals)):
            if fv < f_best:
                f_best = fv
                best_x = p.copy()

        # CMA-ES main loop
        calls_made = len(pop) + 5  # approximate
        for gen in range(200):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Sort population by fitness
            idx = np.argsort(fvals)
            pop_sorted = [pop[i] for i in idx]
            fvals_sorted = [fvals[i] for i in idx]

            # Update mean
            weights = np.ones(lambda_)
            mu = lambda_ // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum = np.sum(w)
            w = w / w_sum
            mu_eff = 1.0 / np.sum(w**2)

            x_old = x.copy()
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * pop_sorted[i]
            x = x_new

            # Update evolution paths
            inv_sqrt_D = 1.0 / np.sqrt(D)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                B @ (D * inv_sqrt_D) @ (x - x_old) / sigma
            )
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (
                1.4 + 2 / (dim + 1)
            ) * np.sqrt(dim):
                pc = (1 - c1) * pc + np.sqrt(
                    c1 * (2 - c1) * mu_eff
                ) * (x - x_old) / sigma

            # Update covariance matrix
            C_old = (B * D) @ B.T
            delta = 1.0
            for i in range(mu):
                delta += (w[i] - 1) * (
                    np.outer(pop_sorted[i] - x, pop_sorted[i] - x)
                )
            C = (1 + c1 + cmu) * C_old - c1 * np.outer(pc, pc) - cmu * delta

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update sigma
            sigma *= np.exp(
                c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1)
            )
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Generate new population
            pop = []
            for _ in range(lambda_):
                z = rng.standard_normal(dim)
                p = x + sigma * (B * D) @ z
                pop.append(np.clip(p, -5, 5))
            fvals = [func(p) for p in pop]
            calls_made += len(pop)

            # Track best
            for p, fv in zip(pop, fvals):
                if fv < f_best:
                    f_best = fv
                    best_x = p.copy()

            # Stagnation check: if no improvement in last 30 generations, restart
            # Simple check: if sigma is very small or hasn't changed much
            if gen > 30 and sigma < 1e-4:
                # Restart from best with larger sigma
                x = best_x.copy()
                sigma = 1.0
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                continue

        return best_x, best_f
