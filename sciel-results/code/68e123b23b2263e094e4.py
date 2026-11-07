"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 280s margin within 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        lambda_pop = 4 + int(3 * np.log(dim))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute Cholesky for sampling (will be updated)
        try:
            C_sqrt_inv = np.linalg.cholesky(C)
            C_sqrt = C_sqrt_inv.T
        except:
            C_sqrt_inv = np.eye(dim)
            C_sqrt = np.eye(dim)

        rng = np.random.default_rng(42)

        while total_calls < budget and time.time() < deadline:
            # 1. Sample population
            pop = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt @ z)
                pop.append(x)
            
            # 2. Evaluate population
            fvals = []
            for x in pop:
                if total_calls >= budget or time.time() > deadline:
                    break
                fvals.append(safe_call(x))
            
            if len(fvals) < lambda_pop:
                break

            # 3. Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # 4. Update weighted mean
            x_new_mean = sum(weights[i] * pop[i] for i in range(mu))

            # 5. Update step size control variable
            if np.linalg.norm(C_sqrt_inv) > 0:
                z_bar = sum(weights[i] * (C_sqrt_inv @ (pop[i] - x_mean) / sigma) for i in range(mu))
            else:
                z_bar = np.zeros(dim)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_bar
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((budget - total_calls) / (c_sigma * (dim + 1))))) < (chi_n + 1e-9) * (1 - c_sigma)
            
            # 6. Update evolution path
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ (x_new_mean - x_mean) / sigma)

            # 7. Update covariance matrix
            delta_sig = 1 + c_sigma * (np.linalg.norm(ps)**2 / chi_n**2 - 1) / 2
            sigma *= np.exp(delta_sig * c_sigma / (dim + 2))
            
            rank_one = np.outer(pc, pc)
            rank_mu = sum(weights[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu)) / sigma**2
            
            C = (1 + c_1 * (hsig - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # 8. Update mean
            x_mean = x_new_mean

            # 9. Update Cholesky decomposition (regularize if needed)
            try:
                C_sqrt_inv = np.linalg.cholesky(C)
                C_sqrt = C_sqrt_inv.T
            except np.linalg.LinAlgError:
                # Regularize C
                C = C + 1e-4 * np.eye(dim)
                try:
                    C_sqrt_inv = np.linalg.cholesky(C)
                    C_sqrt = C_sqrt_inv.T
                except:
                    C = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
