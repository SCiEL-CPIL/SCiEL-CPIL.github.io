import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        eval_history = [] # (x, f)

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            eval_history.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol Sampling
        # Use a power of 2 for Sobol for best properties
        n_samples = 32
        sobol = scipy.stats.qmc.Sobol(d=n, scramble=True)
        # We want points in [-5, 5]^5
        # Sobol generates in [0, 1]^n
        points = sobol.random(n_samples)
        points = points * 10.0 - 5.0
        
        # Evaluate Sobol points
        for p in points:
            try_eval(p)

        # Sort evaluated points by function value
        # eval_history is list of (x, f)
        sorted_points = sorted(eval_history, key=lambda x: x[1])
        
        # Select top 3 distinct starting points for CMA-ES
        # "Distinct" means distance > 1.0 from previously selected
        starts = []
        for x, f in sorted_points:
            if len(starts) >= 3:
                break
            is_distinct = True
            for s in starts:
                if np.linalg.norm(x - s) < 1.0:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(x)

        # If we didn't find enough distinct points, just take the best ones
        while len(starts) < 3 and len(sorted_points) > len(starts):
             # Add next best point even if close
             next_idx = len(starts)
             # This logic is tricky with the list, simpler:
             pass
             
        # Fallback: if starts is empty, use best_x
        if not starts:
            starts = [best_x]

        # Phase 2: CMA-ES from each start
        # Budget per CMA-ES run
        # We have ~470 calls left. 3 runs -> ~150 each.
        cma_budget_per_run = max(50, (budget - total_calls) // max(1, len(starts)) - 10)
        
        def cma_es(x0, max_evals):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return
            
            # CMA-ES Parameters
            lambda_pop = 10
            mu = lambda_pop // 2
            # Weights: linear weights
            weights = np.arange(1, mu + 1)
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initial state
            m = x0.copy()
            sigma = 0.5 # Initial step size
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Precompute Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(n)
            inv_chol_B = np.linalg.inv(B)
            dsigma = 1 + 0.35 * (1 + mu_eff / n)
            
            # Path lengths
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Counter for stagnation
            stagnation_count = 0
            last_best_f = best_f
            last_improvement_iter = 0
            
            for gen in range(max_evals // lambda_pop + 1):
                if total_calls >= budget or time.time() > deadline - 5:
                    return
                
                # Generate population
                pop = np.zeros((lambda_pop, n))
                for i in range(lambda_pop):
                    z = np.random.standard_normal(n)
                    # Transform z using B and sigma
                    pop[i] = m + sigma * (B @ z)
                
                # Evaluate population
                f_vals = np.zeros(lambda_pop)
                for i in range(lambda_pop):
                    f_val = try_eval(pop[i])
                    if f_val is None:
                        return
                    f_vals[i] = f_val
                
                # Sort by function value
                idx = np.argsort(f_vals)
                pop = pop[idx]
                f_vals = f_vals[idx]
                
                # Update mean
                new_m = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update covariance
                # Rank-one update
                v = (new_m - m) / sigma
                c_1_term = c_1 * (np.outer(v, v) - C)
                
                # Rank-mu update
                diff = (pop[:mu] - m) / sigma
                # Weighted outer product
                # w_i * (x_i - m)(x_i - m)^T
                # Using weights
                rank_mu_term = c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', diff, diff), axis=0)
                
                # Update C
                C = (1 - (c_1 + c_mu)) * C + c_1_term + rank_mu_term
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                C += 1e-9 * np.eye(n)
                
                # Update sigma
                # Path history for sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ v)
                sigma = sigma * np.exp((c_sigma / dsigma) * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update path history for C
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (inv_chol_B @ v)
                
                # Hessian correction (simplified: just check if C is PD)
                # In standard CMA-ES, we adjust C to ensure PD.
                # For simplicity and robustness, we rely on the regularization above.
                
                # Update m
                m = new_m
                
                # Check for stagnation
                if f_vals[0] < last_best_f - 1e-8:
                    last_best_f = f_vals[0]
                    last_improvement_iter = gen
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > 5: # 5 generations without improvement
                        break

        # Run CMA-ES for each start
        for s in starts:
            if total_calls >= budget - 10 or time.time() > deadline - 10:
                break
            cma_es(s, cma_budget_per_run)

        # Phase 3: Final Polish (Coordinate Descent)
        if total_calls < budget and time.time() < deadline - 2:
            step = 0.1
            for _ in range(10):
                if total_calls >= budget or time.time() > deadline - 2:
                    break
                improved = False
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 2:
                        break
                    for direction in [-1, 1]:
                        if total_calls >= budget or time.time() > deadline - 2:
                            break
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is not None and f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            improved = True
                if not improved:
                    step *= 0.5
                else:
                    step *= 0.8 # Reduce step if improved, but keep going

        return best_x, best_f
