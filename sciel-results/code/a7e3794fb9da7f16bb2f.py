"""IPOP-CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer with stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # 295s margin within 330s kill limit

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
        lambda_pop = 12  # Smaller pop for faster restarts
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        rng = np.random.default_rng(42)

        # IPOP-CMA-ES state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        try:
            C_sqrt_inv = np.linalg.cholesky(C)
            C_sqrt = C_sqrt_inv.T
        except:
            C_sqrt_inv = np.eye(dim)
            C_sqrt = np.eye(dim)

        # Restart management
        stagnation_count = 0
        best_f_prev = np.inf
        max_stagnation = 10  # Restart if no improvement in 10 generations (120 calls)
        restart_count = 0
        max_restarts = 15  # Allow many restarts within budget
        
        # Track progress for stagnation detection
        best_f_this_run = np.inf

        while total_calls < budget and time.time() < deadline:
            # Check for stagnation and restart
            if stagnation_count > max_stagnation and restart_count < max_restarts:
                restart_count += 1
                # Reset CMA-ES state around the current best point
                x_mean = best_x.copy()
                sigma *= 0.5  # Reduce step size for local refinement
                if sigma < 0.1:
                    sigma = 0.1
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                
                try:
                    C_sqrt_inv = np.linalg.cholesky(C)
                    C_sqrt = C_sqrt_inv.T
                except:
                    C_sqrt_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    
                stagnation_count = 0
                best_f_this_run = best_f
                continue

            # 1. Sample population
            pop = []
            for i in range(lambda_pop):
                if total_calls + i >= budget:
                    break
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                pop.append(x)
            
            if not pop:
                break

            # 2. Evaluate population
            fvals = []
            for x in pop:
                if total_calls >= budget or time.time() > deadline:
                    break
                fvals.append(safe_call(x))
            
            if len(fvals) < lambda_pop:
                # If we couldn't evaluate the full population, break
                if len(fvals) > 0:
                    # Update best if we found something better in partial pop
                    min_idx = np.argmin(fvals)
                    if fvals[min_idx] < best_f:
                        pass # safe_call already updated best_f/best_x
                break

            # 3. Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check stagnation: Did the best in this generation improve the run best?
            if fvals[0] < best_f_this_run - 1e-8:
                best_f_this_run = fvals[0]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # 4. Update weighted mean
            x_new_mean = sum(weights[i] * pop[i] for i in range(mu))

            # 5. Update step size control variable
            z_bar = sum(weights[i] * (C_sqrt_inv @ (pop[i] - x_mean) / sigma) for i in range(mu))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_bar
            
            # Check for path norm
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(total_calls / (c_sigma * (dim + 1))))) < (chi_n + 1e-9) * (1 - c_sigma)
            
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

            # 9. Update Cholesky decomposition
            try:
                C_sqrt_inv = np.linalg.cholesky(C)
                C_sqrt = C_sqrt_inv.T
            except np.linalg.LinAlgError:
                C = C + 1e-4 * np.eye(dim)
                try:
                    C_sqrt_inv = np.linalg.cholesky(C)
                    C_sqrt = C_sqrt_inv.T
                except:
                    C = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    # Force a restart if matrix is totally broken
                    stagnation_count = max_stagnation + 1

        return best_x, best_f
# EVOLVE-BLOCK-END
