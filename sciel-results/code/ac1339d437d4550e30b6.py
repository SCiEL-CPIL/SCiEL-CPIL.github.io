"""Optimizer for BBOB-style f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # Try multiple restarts with different initializations
        # Each restart gets a portion of the budget
        # We'll do adaptive restarts: if we stagnate, restart with new center

        def cma_es(x0, f0, budget_for_phase, rng_local):
            """Run CMA-ES from x0 with f0 for budget_for_phase evaluations."""
            n = self.dim
            # CMA-ES parameters
            # Population size: 4 + floor(3 * log(n))
            lambda_ = 4 + int(3 * np.log(n))
            # Standard deviation
            sigma = 2.0
            # Step size adaptation (CSA)
            # For simplicity, use a simplified CMA-ES with step size adaptation
            # based on success rate (1/5 rule generalized)

            x = x0.copy()
            fx = f0
            best_x = x0.copy()
            best_f = f0
            calls = 1

            # Covariance matrix (identity initially)
            C = np.eye(n)
            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Parameters
            mu = lambda_ // 2  # number of parents
            weights = np.ones(mu)
            # Standard CMA-ES weights: w_i = log(mu+0.5) - log(i)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Adaptation rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((n + 2) ** 2 + mu_eff))

            # Hessian of the log determinant of C (approximate)
            # For simplicity, we'll use a simplified update

            for _ in range(budget_for_phase - calls):
                if time.time() > deadline:
                    break

                # Generate population
                Y = np.zeros((lambda_, n))
                for i in range(lambda_):
                    # Sample from N(0, C)
                    # Use Cholesky decomposition
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    Y[i] = x + sigma * (L @ rng_local.standard_normal(n))
                    Y[i] = np.clip(Y[i], -5, 5)

                # Evaluate population
                F = np.zeros(lambda_)
                for i in range(lambda_):
                    F[i] = func(Y[i])
                    calls += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = Y[i].copy()

                # Sort by fitness
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]

                # Mean update
                x_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Step size adaptation (CSA)
                y = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / np.sqrt(mu_eff)) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(mu_eff)))
                sigma = max(sigma, 1e-12)

                # Covariance matrix adaptation
                if h_sigma:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                else:
                    p_c = (1 - c_c) * p_c

                v = np.outer(p_c, p_c)
                C = (1 + c_1 * (mu_eff - 1) - c_mu) * C + c_1 * v
                # Rank-mu update
                z = np.zeros((mu, n))
                for i in range(mu):
                    z[i] = (Y[i] - x) / sigma
                C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', weights, z, z), axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Update mean
                x = x_new

            return best_x, best_f, calls

        # Phase 1: Initial exploration with larger sigma
        # Start from center
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        global_best_x = x0.copy()
        global_best_f = f0

        # Use CMA-ES with restarts
        # Allocate budget across phases
        # Phase 1: 60% of budget
        # Phase 2: 25% of budget (restart if needed)
        # Phase 3: 15% of budget (final refinement)

        phase_budgets = [int(self.budget * 0.5), int(self.budget * 0.3), int(self.budget * 0.2)]
        # Adjust to match total budget
        phase_budgets[-1] = self.budget - phase_budgets[0] - phase_budgets[1]

        for phase_idx, pb in enumerate(phase_budgets):
            if time.time() > deadline:
                break

            if phase_idx == 0:
                x_start = np.zeros(self.dim)
                f_start = global_best_f
            else:
                # Restart from best found so far with some perturbation
                x_start = global_best_x + rng.uniform(-0.5, 0.5, self.dim)
                x_start = np.clip(x_start, -5, 5)
                f_start = func(x_start)
                if f_start < global_best_f:
                    global_best_f = f_start
                    global_best_x = x_start.copy()

            # Run CMA-ES for this phase
            best_x, best_f, _ = cma_es(x_start, f_start, pb, rng)
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
