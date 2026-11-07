"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time-guarded budget."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        dim = self.dim
        total_budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        # Phase 1: Broad random sampling to find promising basins
        # Weierstrass f16 has many local minima; broad search helps
        n_initial = 50
        for i in range(n_initial):
            if calls >= total_budget or not time_ok():
                break
            x = np.random.default_rng(i * 1000 + 42).uniform(-4, 4, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None

        # Phase 2: CMA-ES refinement from the best starting point
        # Use a simplified CMA-ES implementation
        x = best_x.copy()
        f = best_f
        sigma = 1.0  # initial step size

        # CMA-ES parameters
        n = dim
        lambda_ = 4 * n  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = (2 - c_sigma + c_c) / ((n + 1.3) ** 2 + c_sigma)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + c_sigma))

        # Initialize CMA-ES state
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track progress to decide when to restart
        stall_count = 0
        stall_threshold = 50  # restart if no improvement for this many iterations
        last_best_f = f
        last_improve_iter = 0

        while calls < total_budget and time_ok():
            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls + i >= total_budget or not time_ok():
                    break
                z = np.random.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                fy = func(np.clip(y, -5, 5))
                calls += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < f:
                f = offspring[0][0]
                x = offspring[0][1].copy()
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                last_improve_iter = calls
                stall_count = 0
            else:
                stall_count += 1

            # Check for stagnation - restart with new center if stuck
            if stall_count >= stall_threshold and calls < total_budget - 100:
                # Try a different starting point
                rng_restart = np.random.default_rng(calls)
                new_x = rng_restart.uniform(-4, 4, n)
                new_f = func(new_x)
                calls += 1
                if new_f < f:
                    x = new_x.copy()
                    f = new_f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                sigma = 1.0
                B = np.eye(n)
                D = np.ones(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                stall_count = 0
                continue

            # Update CMA-ES parameters
            w = weights[:len(offspring)]
            m_new = np.sum(w[:, np.newaxis] * np.array([o[1] for o in offspring]), axis=0)

            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * w @ (x - m_new) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * w @ (x - m_new) / sigma

            # Update covariance matrix
            delta_h = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - last_improve_iter)))) <= (chi_n + 1.4)
            delta_h = 1.0  # Simplified: always update
            C_update = (1 + c_1 * delta_h - c_mu) * np.outer(pc, pc)
            C_update += c_mu * (np.outer(w @ (x - m_new) / sigma, w @ (x - m_new) / sigma))
            C_update = C_update + c_1 * (np.eye(n) - np.outer(pc, pc))

            # Ensure positive definite
            C_update = 0.5 * (C_update + C_update.T)
            eigvals, eigvecs = np.linalg.eigh(C_update)
            eigvals = np.maximum(eigvals, 1e-10)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update mean
            x = m_new

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-4, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
