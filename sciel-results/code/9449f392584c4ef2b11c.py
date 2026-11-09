import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin from 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Strategy: Multi-start CMA-ES
        # CMA-ES is robust to rotations and shifts, which are present in the problem.
        # We use a fixed population size and restart if stagnation is detected or budget allows.
        
        # Parameters for CMA-ES
        # Standard lambda for dim=5 is often 4 + floor(3*log(dim)) ~ 14, but 10 is a good balance for budget 500.
        # Let's use lambda = 10 for efficiency.
        lam = 10
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Initial standard deviation
        # The search space is [-5, 5]^5. A reasonable initial sigma is 0.5 to 1.0.
        # Let's start with 0.5 to allow some global exploration but focus on local structure.
        
        # We will run multiple CMA-ES instances (restarts) if the first one stagnates or finishes.
        # Given budget=500, and lam=10, we can afford ~50 generations per run.
        # A single run might not be enough to find the global optimum if the initial guess is bad.
        # So, we split the budget or use a stagnation-based restart.
        
        # Approach:
        # 1. Run CMA-ES with a portion of the budget (e.g., 60%).
        # 2. If it stagnates (sigma becomes very small or no improvement for X generations), restart with new random center.
        # 3. Use remaining budget for a final local polish around the global best.
        
        # To keep it simple and robust, let's implement a single CMA-ES run that adapts, 
        # but with a "restart" logic if progress is too slow.
        
        # Actually, a simpler and very effective strategy for this specific "black-box, rotated, shifted" scenario 
        # is a single CMA-ES run with a sufficiently large initial sigma and good stopping criteria.
        
        # Let's implement a standard CMA-ES.
        
        def run_cma_es(initial_mean, initial_sigma, budget_for_run):
            nonlocal calls_used, best_f, best_x
            
            # CMA-ES State
            mean = np.array(initial_mean)
            sigma = initial_sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Step sizes
            # cm = (mueff + 2) / (dim + mueff)
            # cs = (mueff + 1.5) / (dim + mueff + 3)
            # cc = (2 + mueff/dim) / (dim + 2 + mueff/dim)
            # cmu = mueff * weights[0] * (2 - mueff/dim) / (dim + 2) # Simplified, let's use standard values
            
            cm = (mueff + 2) / (dim + mueff)
            cs = (mueff + 1.5) / (dim + mueff + 3)
            cc = (2 + mueff / dim) / (dim + 2 + mueff / dim)
            
            # Precompute cholesky factor update
            # We will recompute Cholesky every generation for simplicity (dim=5 is small)
            
            gen = 0
            stagnation_counter = 0
            last_best_f = best_f
            
            while calls_used < budget_for_run and time.time() < deadline - 5.0:
                # Generate population
                pop = np.zeros((lam, dim))
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes singular
                    L = np.eye(dim)
                    
                for i in range(lam):
                    z = np.random.randn(dim)
                    pop[i] = mean + sigma * (L @ z)
                    # Clip to bounds? CMA-ES usually handles bounds via penalty or repair.
                    # The problem says func clips. So we can let CMA-ES propose points outside,
                    # but it's better to clip the proposal to stay within the box to avoid wasted evaluations?
                    # The contract says "func(x) clips a finite length-5 vector to the box".
                    # So we can pass unclipped points, but clipping might be more efficient if we know the box.
                    # Let's clip to [-5, 5] to keep the search focused.
                    pop[i] = np.clip(pop[i], -5, 5)
                
                # Evaluate population
                fit = np.zeros(lam)
                for i in range(lam):
                    if calls_used >= budget_for_run:
                        break
                    fit[i] = track_call(pop[i])
                
                # Sort by fitness
                idx = np.argsort(fit)
                pop = pop[idx]
                fit = fit[idx]
                
                # Update mean
                new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                # Update path
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ ((new_mean - mean) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mueff + 1)
                
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (new_mean - mean) / sigma
                
                # Update Covariance Matrix
                delta_sig = (np.linalg.norm(ps) ** 2 / dim - 1) / (1 - (1 - cs)**(2 * (gen + 1)))
                C = (1 - cm) * C + cm * (np.outer(pc, pc) + delta_sig * C)
                
                # Ensure symmetry and positive definiteness (simple regularization)
                C = (C + C.T) / 2
                # Add small regularization if needed
                # C += 1e-12 * np.eye(dim)
                
                mean = new_mean
                
                # Adapt sigma
                sigma = sigma * np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                gen += 1
                
                # Stagnation check
                if best_f < last_best_f - 1e-8:
                    last_best_f = best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # If stagnation is high and we have budget, we might break to restart, 
                # but for now, let's just run until budget or deadline.
                # A better strategy for multi-start would be to break here if stagnation_counter > 10.
                
            return best_x, best_f

        # Phase 1: Global Search with CMA-ES
        # Start with a random point in the center or a random point?
        # Random point is safer for "hidden shift".
        start_x = np.random.uniform(-3, 3, dim)
        
        # Use 70% of budget for CMA-ES
        cma_budget = int(budget * 0.7)
        run_cma_es(start_x, 0.5, cma_budget)
        
        # Phase 2: Local Refinement
        # Use remaining budget for a high-precision local search.
        # We can use a very small sigma CMA-ES or Nelder-Mead.
        # Let's use a small sigma CMA-ES for consistency and robustness.
        
        if calls_used < budget and time.time() < deadline - 5.0:
            remaining = budget - calls_used
            if remaining > 10:
                # Restart CMA-ES with small sigma around best_x
                run_cma_es(best_x, 0.01, budget)
        
        return best_x, best_f
