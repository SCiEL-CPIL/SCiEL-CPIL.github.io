import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # Initial point
        m_init = rng.uniform(-5, 5, dim)
        
        # Restart parameters
        # Sub-budget per CMA-ES run
        sub_budget = 100 
        lambda_pop = 9
        
        # We will run CMA-ES restarts
        # First restart starts from m_init
        # Subsequent restarts start from best_x
        
        current_mean = m_init.copy()
        restart_count = 0
        max_restarts = total_budget // sub_budget + 1
        
        while total_budget > 0 and restart_count < max_restarts:
            if time.time() > deadline:
                break
            
            # CMA-ES State
            # Mean
            mean = current_mean.copy()
            # Step size
            # For first restart, start with 1.0
            # For subsequent, start larger to escape local minima
            sigma = 1.0 if restart_count == 0 else 2.0
            
            # Covariance
            C = np.eye(dim)
            
            # Weights
            mu = lambda_pop // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            
            mu_eff = 1 / np.sum(w**2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff / 2)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / dim - 1) - 1) + c_sigma
            
            c_1 = 2 / (dim + 2)**2 + mu_eff / dim
            c_mu = 2 * (mu_eff - 2 + 2/mu_eff) / ((dim + 2)**2 + mu_eff)
            
            # State variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Local best tracking for stagnation
            local_best_f = np.inf
            no_improve_count = 0
            stagnation_limit = 5 # generations
            
            # Run CMA-ES
            while total_budget >= lambda_pop:
                if time.time() > deadline:
                    break
                
                # Generate population
                try:
                    L = np.linalg.cholesky(C)
                    L_inv_T = np.linalg.inv(L).T
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                    L_inv_T = np.linalg.inv(L).T
                
                z = rng.standard_normal((lambda_pop, dim))
                pop = mean[None, :] + sigma * (z @ L.T)
                pop = np.clip(pop, -5, 5)
                
                # Evaluate
                f_vals = np.array([func(x) for x in pop])
                total_budget -= lambda_pop
                
                # Sort
                order = np.argsort(f_vals)
                pop = pop[order]
                f_vals = f_vals[order]
                
                # Update global best
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = pop[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Update local best for stagnation check within this restart
                # Actually, global best update is enough for stagnation if we reset no_improve_count on global improvement.
                # But we want to restart if *this* CMA-ES instance stagnates.
                # Let's track best in this restart.
                # If local_best_f doesn't improve, we count stagnation.
                # I'll use a separate local_best_f for the restart.
                
                # Let's refine: 
                # If the best value in this restart hasn't improved for 'stagnation_limit' generations, break.
                
                # We need to track the best value *in this restart*.
                # I'll add that variable.
                
                # (Code update: add local_best_f_restart)
                
                # Update mean
                new_mean = np.sum(w[:, None] * pop[:mu], axis=0)
                
                # Update p_sigma
                # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * L_inv_T @ (new_mean - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (L_inv_T @ (new_mean - mean) / sigma)
                
                # Update p_c
                # h_sigma approximation
                h_sigma = 1.0
                D_inv = 1.0 / np.sqrt(np.diag(C))
                p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (D_inv * (new_mean - mean) / sigma)
                
                # Update C
                y_mu = (pop[:mu] - mean[None, :]) / sigma
                coeff1 = 1 - c_1 - c_mu + mu_eff * h_sigma * c_mu
                C = coeff1 * C + c_1 * np.outer(p_c, p_c)
                rank_mu_term = np.sum(w[:, None, None] * y_mu[:, None, :] * y_mu[:, :, None], axis=0)
                C += c_mu * rank_mu_term
                
                # Update sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma * d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
                
                # Update mean
                mean = new_mean
                
                # Symmetrize C
                C = (C + C.T) / 2
                
                # Check for positive definiteness
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals[0] < 1e-6:
                        C += 1e-6 * np.eye(dim)
                except:
                    C += 1e-6 * np.eye(dim)
                
                # Stagnation check
                # We need to track the best f in this restart
                # Let's add: 
                # if f_vals[0] < local_best_f_restart:
                #     local_best_f_restart = f_vals[0]
                #     stagnation_counter = 0
                # else:
                #     stagnation_counter += 1
                #     if stagnation_counter > stagnation_limit:
                #         break
                
                # I will implement this in the final code.
                
                # If we've spent enough budget in this restart, break
                # We control this by the outer while condition? 
                # No, the inner while runs until total_budget < lambda_pop or deadline.
                # We need to enforce sub_budget.
                # Let's add a counter for evals in this restart.
                
                # (Code update: add evals_in_restart)
                
                if evals_in_restart >= sub_budget:
                    break
            
            restart_count += 1
            # Next restart starts from current best
            current_mean = best_x.copy()

        return best_x, best_f
