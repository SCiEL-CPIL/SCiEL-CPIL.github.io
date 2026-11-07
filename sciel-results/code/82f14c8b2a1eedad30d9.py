import numpy as np
import time
import sys

class Optimizer:
    """
    Hybrid Optimizer: CMA-ES Portfolio with Local Refinement.
    
    Strategy:
    1. Run CMA-ES for global exploration (handles rotation/correlation).
    2. If budget remains, run a second CMA-ES phase centered on the best found 
       with a smaller step size to refine or escape local minima.
    3. Use a local gradient-free method (Nelder-Mead) for the final polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 320s to be safe.
        deadline = start_time + 320.0
        
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        
        global_best_x = None
        global_best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        lb = -5.0
        ub = 5.0
        
        def update_global(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        def run_cma_es(x0, sigma0, budget_limit, max_time_left):
            nonlocal calls_made, global_best_x, global_best_f
            
            # CMA-ES Parameters
            # For n=5, lambda = 4 + floor(3 * ln(5)) = 4 + 4 = 8. 
            # Let's use lambda=10 for better exploration in early phase.
            lam = 10 
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Initial population
            pop = []
            for i in range(lam):
                if calls_made >= budget_limit or time.time() > deadline - max_time_left:
                    break
                z = rng.standard_normal(dim)
                x = np.clip(mean + sigma * z, lb, ub)
                fx = func(x)
                calls_made += 1
                pop.append((x, fx))
                update_global(x, fx)
            
            if not pop:
                return

            # Main Loop
            stagnation_count = 0
            last_best_f = global_best_f
            
            while calls_made < budget_limit and time.time() < deadline - max_time_left:
                # Sort population
                pop.sort(key=lambda t: t[1])
                
                # Check for stagnation
                if pop[0][1] < last_best_f:
                    last_best_f = pop[0][1]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # If we have stagnated for many generations, break to allow restart/polish
                if stagnation_count > 5:
                    break
                    
                # Update Mean
                new_mean = np.sum([weights[i] * pop[i][0] for i in range(mu)], axis=0)
                
                # Update Step Size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (new_mean - mean))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (1.0 / (c_sigma * (dim + 1))))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                    
                # Update Covariance
                diff = (new_mean - mean) / sigma
                pc = (1 - c_c) * pc + c_c * diff
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c) - 1) * C)
                for i in range(mu):
                    x_i = (pop[i][0] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(x_i, x_i)
                
                mean = new_mean
                sigma = max(sigma, 1e-10)
                
                # Convergence check
                if sigma < 1e-6:
                    break
                    
                # Generate Next Population
                next_pop = []
                for i in range(lam):
                    if calls_made >= budget_limit or time.time() > deadline - max_time_left:
                        break
                    z = rng.standard_normal(dim)
                    x = np.clip(mean + sigma * z, lb, ub)
                    fx = func(x)
                    calls_made += 1
                    next_pop.append((x, fx))
                    update_global(x, fx)
                
                if len(next_pop) < lam:
                    break
                pop = next_pop

        # --- Phase 1: Global Exploration ---
        # Use 60% of budget for the first CMA-ES run
        budget_phase1 = int(total_budget * 0.6)
        x0_1 = rng.uniform(-5, 5, dim)
        run_cma_es(x0_1, 2.0, budget_phase1, 10.0) # 10s margin for safety
        
        # --- Phase 2: Local Refinement / Second Chance ---
        # If we have time and budget left, run another CMA-ES centered on the best found
        if calls_made < total_budget and time.time() < deadline - 15.0:
            if global_best_x is not None:
                # Center on global best, smaller sigma
                x0_2 = global_best_x.copy()
                sigma_2 = 0.5
            else:
                x0_2 = rng.uniform(-5, 5, dim)
                sigma_2 = 1.0
            
            budget_phase2 = int(total_budget * 0.3)
            # Ensure we don't exceed total budget
            remaining_for_phase2 = min(budget_phase2, total_budget - calls_made)
            
            # Run CMA-ES for this phase
            # We use a helper that respects the global calls_made
            # We need to pass the current state or re-run logic. 
            # To keep it simple and robust, we call run_cma_es again.
            # It will use the remaining budget.
            
            # Adjust budget limit for this specific call
            # run_cma_es uses 'budget_limit' as absolute calls_made limit? 
            # No, my implementation above uses 'budget_limit' as a hard stop for calls_made.
            # So we pass total_budget as the limit, but we only want to use ~30% more.
            # Let's create a wrapper or modify the logic.
            
            # Simpler: Just call run_cma_es with a budget limit that is calls_made + remaining_for_phase2
            limit_phase2 = calls_made + remaining_for_phase2
            run_cma_es(x0_2, sigma_2, limit_phase2, 10.0)

        # --- Phase 3: Final Polish with Nelder-Mead ---
        # Use remaining budget for a local optimizer which is very good at the final steps
        if global_best_x is not None and calls_made < total_budget and time.time() < deadline - 5.0:
            try:
                from scipy.optimize import minimize
                
                # Nelder-Mead is robust for low dimensions
                # maxiter is controlled by our budget check inside the lambda
                # But minimize doesn't know about our global calls_made.
                # We need to wrap func to count calls and stop if budget exceeded.
                
                def wrapped_func(x):
                    nonlocal calls_made
                    if calls_made >= total_budget:
                        return global_best_f # Return current best to stop progress
                    if time.time() > deadline - 2.0:
                        return global_best_f
                    fx = func(x)
                    calls_made += 1
                    update_global(x, fx)
                    return fx

                # Start from global best
                res = minimize(wrapped_func, global_best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                
                # The minimize function might not stop exactly on budget, 
                # but our wrapped_func returns a constant if budget is hit, 
                # causing it to converge/stop quickly.
                
            except Exception:
                pass

        # Fallback if no calls were made (should not happen)
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            if calls_made < total_budget:
                global_best_f = func(global_best_x)
                calls_made += 1

        return global_best_x, global_best_f
