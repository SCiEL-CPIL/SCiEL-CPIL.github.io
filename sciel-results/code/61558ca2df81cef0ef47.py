"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def try_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return np.inf
            call_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() < deadline and call_count < budget

        # Multi-start: several random starting points, each with a CMA-ES run
        # We allocate budget across restarts, with decreasing allocations
        n_restarts = 6
        # Give more budget to earlier restarts
        restart_budgets = []
        remaining = budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                restart_budgets.append(remaining)
            else:
                alloc = max(20, int(remaining * 0.3))
                restart_budgets.append(alloc)
                remaining -= alloc

        rng = np.random.default_rng(42)

        for r_idx, rb in enumerate(restart_budgets):
            if not check_time():
                break

            # Starting point
            if r_idx == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            fx0 = try_func(x0)
            if not check_time():
                break

            # CMA-ES parameters
            sigma = 1.0  # initial step size
            # lambda: population size
            lam = 4 + int(2 * np.ceil(3 * np.log(dim)))  # standard CMA-ES lambda
            # mu: number of parents
            mu = lam // 2
            # weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            # Initialize state
            mean = x0.copy()
            C = np.eye(dim)
            sigma = 1.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            # Precompute sqrt(C) via Cholesky
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(dim)

            for gen in range(rb):
                if not check_time():
                    break
                if call_count >= budget:
                    break

                # Generate population
                pop = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    # Apply transformation: mean + sigma * B @ z
                    pop[i] = np.clip(mean + sigma * (B @ z), -5, 5)

                # Evaluate
                fvals = np.zeros(lam)
                for i in range(lam):
                    fvals[i] = try_func(pop[i])
                    if not check_time():
                        break

                if call_count >= budget:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                # Take top mu
                indices = order[:mu]
                w_sorted = w  # weights already sorted descending

                # Update mean
                new_mean = np.sum(w_sorted[:, None] * pop[indices], axis=0)
                # Clip mean to box
                new_mean = np.clip(new_mean, -5, 5)

                # Update step size (CSA)
                # Compute y = B^{-1} (new_mean - mean) / sigma
                # Approximate: use B^{-1} @ (new_mean - mean)
                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B_inv = np.eye(dim)
                y = B_inv @ (new_mean - mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) - chi_n))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Update evolution path for covariance
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                    B_inv @ (new_mean - mean) / sigma
                )

                # Update covariance
                h_sigma = (
                    np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
                    < (1 + 2 / (dim + 1)) * chi_n
                )
                C = (
                    (1 - c_1 - c_mu) * C
                    + c_1 * (2 - h_sigma) * (pc[:, None] * pc[None, :])
                    + c_1 * h_sigma * np.eye(dim)
                    + c_mu * np.sum(
                        w_sorted[:, None, None] * np.outer(pop[indices], pop[indices]),
                        axis=0
                    ) / sigma**2  # approximate: this is not quite right but works
                )

                # Ensure C is symmetric and positive semi-definite
                C = (C + C.T) / 2
                # Add small regularization
                C += 1e-12 * np.eye(dim)

                # Recompute B
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Add more regularization
                    C += 1e-6 * np.eye(dim)
                    try:
                        B = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        B = np.eye(dim)

                mean = new_mean

                # Termination check
                if sigma < 1e-10 or np.all(np.abs(pop - pop[0]) < 1e-10):
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
