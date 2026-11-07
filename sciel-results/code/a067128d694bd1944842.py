import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            # Clip to box [-5, 5] as per problem description
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # --- CMA-ES with Multi-Start and Adaptive Restart ---
        
        # We will run multiple independent CMA-ES instances.
        # Each instance gets a fraction of the budget.
        # If an instance stagnates, we restart it or move to the next.
        
        # Strategy:
        # 1. Initial broad exploration with a few random points to find a good basin.
        # 2. Run CMA-ES starting from the best point found.
        # 3. If CMA-ES stagnates (sigma too small or no improvement), restart from a new random point.
        
        # Let's define a helper to run one CMA-ES instance with a given budget
        def run_cmaes(start_mean, start_sigma, max_calls, func_eval):
            # Parameters for dim=5
            lambda_pop = 10
            mu = lambda_pop // 2
            
            # Weights
            weights = np.log(lambda_pop / 2.0 + 1.0) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2.0) / (lambda_pop + mu_eff + 5.0)
            c_c = (4.0 + mu_eff / lambda_pop) / (lambda_pop + 4.0 + 2.0 * mu_eff / lambda_pop)
            c_1 = 2.0 / (lambda_pop**2 + 6.0)
            c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (lambda_pop**2 + 2.0))
            d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) * (lambda_pop - mu_eff - 2.0) / (mu_eff + 1.0) - 1.0)) + c_sigma
            
            # Initial state
            mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(dim)
            C_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_best_f = np.inf
            local_best_x = start_mean.copy()
            no_improve_count = 0
            
            def sample_from_mvnormal(mean, sigma, C):
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                z = np.random.standard_normal((lambda_pop, dim))
                return mean + sigma * (z @ L.T)
            
            # Main Loop for this instance
            while calls < budget and time_ok():
                if calls + lambda_pop > budget:
                    # If we don't have enough budget for a full population, break
                    # Or we could do a partial population, but let's keep it simple
                    # Actually, we can just check if we have at least 1 call left
                    if calls >= budget:
                        break
                    
                # 1. Sample population
                pop = sample_from_mvnormal(mean, sigma, C)
                
                # 2. Evaluate
                fitness = np.array([func_eval(x) for x in pop])
                
                # Check for NaN or Inf
                valid_mask = np.isfinite(fitness)
                if not np.any(valid_mask):
                    break
                    
                # 3. Sort by fitness (only valid ones, but we assume all are finite due to func returning float)
                # The problem says func returns a float, so we assume no NaNs unless the function itself returns them.
                # The contract says "returns a float", implying valid floats.
                order = np.argsort(fitness)
                pop = pop[order]
                fitness = fitness[order]
                
                # Update local best
                if fitness[0] < local_best_f:
                    local_best_f = fitness[0]
                    local_best_x = pop[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Stagnation check: if no improvement for many generations, restart might be better
                # But we handle restart at the outer level.
                # If sigma is too small, we can also stop this instance.
                if sigma < 1e-12:
                    break
                
                # 4. Update mean
                mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                # 5. Update covariance matrix
                y = pop[:mu] - mean
                sum_yi_yi = np.sum(weights[:, None, None] * y[:, :, None] * y[:, None, :], axis=0)
                
                # Update path cm
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * sigma * (C_inv @ (mean_new - mean))
                
                # Update sigma
                e_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
                norm_pc = np.linalg.norm(pc)
                sigma *= np.exp(c_sigma / d_sigma * (norm_pc / (np.sqrt(mu_eff) * e_norm) - 1.0))
                
                # Update C
                C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1 - c_1) * d_sigma) * np.outer(pc, pc) + c_mu * sum_yi_yi
                C = (C + C.T) / 2.0
                
                # Update C_inv
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C_inv = np.linalg.inv(C + 1e-10 * np.eye(dim))
                    
                mean = mean_new
                
            return local_best_x, local_best_f

        # --- Multi-Start Strategy ---
        
        # Phase 1: Random Sampling to find a good starting point
        # Use ~50 evaluations for random sampling
        num_random = min(50, budget)
        random_points = np.random.uniform(-5, 5, size=(num_random, dim))
        for x in random_points:
            f = eval_func(x)
        
        # Phase 2: Run CMA-ES from the best point found
        # We have budget - num_random calls left
        remaining_budget = budget - calls
        
        # Run CMA-ES with the remaining budget
        # We pass a wrapper to eval_func to respect the global budget and time
        # But run_cmaes uses the global `calls` and `budget` via closure?
        # No, run_cmaes takes `func_eval` which should be `eval_func`.
        # However, run_cmaes has its own loop condition `calls < budget`.
        # We need to ensure it stops when global budget is reached.
        # The `eval_func` already checks `calls >= budget` and returns np.inf.
        # The loop in run_cmaes checks `calls < budget`.
        # So it should be safe.
        
        # Start CMA-ES from the global best
        # Initial sigma: 1.0 is a reasonable guess for the scale of the remaining search
        # Or we can use a larger sigma if we want more exploration.
        # Let's start with sigma = 1.0
        
        best_x_cma, best_f_cma = run_cmaes(best_x, 1.0, remaining_budget, eval_func)
        
        # If CMA-ES didn't improve, or if we have time/budget left, we could try another start.
        # But with 500 calls, 50 random + 450 CMA-ES is a good split.
        # CMA-ES in 5D with 450 calls (45 generations of 10) is quite powerful.
        
        # If we have more time, we could do a second CMA-ES from a different random point.
        # But let's stick to one strong CMA-ES run for now to maximize convergence.
        
        return best_x, best_f
