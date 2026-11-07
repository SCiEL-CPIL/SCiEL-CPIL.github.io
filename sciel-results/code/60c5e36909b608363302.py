"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for the BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, budget, dim, rng, start_mean, start_sigma, deadline):
        """Run a single CMA-ES instance and return (best_x, best_f, evals_used)."""
        # CMA-ES Parameters
        # Standard lambda for D=5 is 4 + floor(3 * log(2*D)) = 4 + floor(3*2.58) = 4+7=11, or simply 9+3*floor(log10(4*D))
        # Let's use a standard lambda = 9 + 3 * floor(log10(4 * dim))
        lambda_pop = 9 + 3 * int(np.floor(np.log10(4 * dim)))
        if lambda_pop < 10:
            lambda_pop = 10

        mu = lambda_pop // 2
        # Weights: w_i = ln(mu + 0.5) - ln(i) for i in 1..mu
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)

        mu_eff = 1.0 / np.sum(weights**2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initialization
        mean = start_mean.copy()
        sigma = start_sigma
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute Cholesky of C (initially Identity)
        C_sqrt = np.eye(dim)

        best_x = mean.copy()
        best_f = np.inf

        evals = 0

        # Helper to check time
        def time_exceeded():
            return time.time() > deadline

        # Main Loop
        while evals < budget and not time_exceeded():
            # Generate population
            pop = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                if time_exceeded():
                    break
                # Sample from Normal(mean, sigma^2 C)
                # y ~ N(0, I), x = mean + sigma * C_sqrt @ y
                y = rng.standard_normal(dim)
                z = C_sqrt @ y
                pop[i] = mean + sigma * z
                # Clip to box [-5, 5]
                np.clip(pop[i], -5, 5, out=pop[i])
                evals += 1

            if time_exceeded():
                break

            # Evaluate fitness
            fitness = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if evals > budget:
                    break
                try:
                    f_val = func(pop[i])
                except Exception:
                    f_val = np.inf
                fitness[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = pop[i].copy()

            if time_exceeded():
                break

            # Sort population by fitness
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]

            # Update Mean
            # w is defined for the first mu individuals
            mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)

            # Calculate step for CSA
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * C^-1/2 * (mean_new - mean) / sigma
            # We need C^-1/2. We have C_sqrt = C^1/2. So C^-1/2 = inv(C_sqrt).
            # To avoid explicit inverse, we can solve C_sqrt @ v = (mean_new - mean) / sigma
            delta_mean = (mean_new - mean) / sigma
            try:
                v = np.linalg.solve(C_sqrt, delta_mean)
            except np.linalg.LinAlgError:
                v = delta_mean  # Fallback

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^-1/2 * (mean_new - mean) / sigma
            # Note: The standard CMA-ES update for ps uses the same v as above?
            # Yes, ps is the evolution path in the search space (transformed by C^-1/2).
            # Actually, standard formula:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^-1/2 * (x_new - x_old) / sigma
            # Here (x_new - x_old) is (mean_new - mean).
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v

            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1-c_c)^2) * np.eye(dim)) 
            #    + c_mu * sum(w_i * (y_i @ y_i.T)) where y_i = (x_i - mean_old)/sigma
            # Note: The rank-mu update uses the selected individuals.
            # Let's compute the rank-1 part and rank-mu part.

            # Rank-1 part
            # The term (1 - (1-c_c)^2) * I is added to pc @ pc.T in the standard formula?
            # Formula: C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1-c_c)^2) * I) + c_mu * H
            # where H = sum_{i=1}^mu w_i * y_i @ y_i.T

            # Calculate H
            H = np.zeros((dim, dim))
            for i in range(mu):
                # y_i = (pop[i] - mean) / sigma  (Using old mean)
                y_i = (pop[i] - mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)

            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(dim))
            C += c_mu * H

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2

            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, add small regularization
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break  # Give up if matrix becomes unstable

            # Update mean
            mean = mean_new

            # Termination check: if sigma is very small
            if sigma < 1e-10:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, stop at 290s

        dim = self.dim
        budget = self.budget

        # We will use a multi-start strategy with CMA-ES.
        # The total budget is 500. We can split it into several runs.
        # Let's try 5 runs of 100 evals each.
        # Or maybe adaptive: if the first run converges quickly, use the rest for more.
        # For simplicity and robustness, let's do fixed restarts.
        # 500 evals / 5 restarts = 100 evals per restart.
        # This allows escaping local minima.

        n_restarts = 5
        budget_per_restart = budget // n_restarts

        best_x_global = None
        best_f_global = np.inf

        rng = np.random.default_rng(0)

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Choose starting point
            # First run: center (0,0,0,0,0)
            # Subsequent runs: random points, or perturbations of the best found so far
            if r == 0:
                start_mean = np.zeros(dim)
                start_sigma = 2.0
            else:
                # Perturb the best found so far
                if best_x_global is not None:
                    # Random perturbation around the best
                    perturb_scale = 1.0
                    start_mean = best_x_global + perturb_scale * rng.standard_normal(dim)
                    start_mean = np.clip(start_mean, -5, 5)
                    start_sigma = 1.0
                else:
                    start_mean = rng.uniform(-5, 5, dim)
                    start_sigma = 2.0

            # Run CMA-ES
            try:
                best_x, best_f, evals_used = self._cma_es_run(
                    func, budget_per_restart, dim, rng, start_mean, start_sigma, deadline
                )
            except Exception:
                break

            if best_f < best_f_global:
                best_f_global = best_f
                best_x_global = best_x

        # If no global best was found (shouldn't happen), fallback
        if best_x_global is None:
            # Just evaluate a few random points
            for _ in range(min(10, budget)):
                x = rng.uniform(-5, 5, dim)
                try:
                    f = func(x)
                except Exception:
                    f = np.inf
                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
