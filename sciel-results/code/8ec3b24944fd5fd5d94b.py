"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start CMA-ES with adaptive restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        
        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        
        # CMA-ES parameters
        # Population size heuristic: 4 + floor(3 * log(n))
        lambda_pop = int(4 + 3 * np.log(n))
        # Initial sigma: heuristic based on domain size [-5, 5]
        # For Katsuura, local minima are dense. Starting with a moderate sigma.
        # If we use multi-start, we can start with a smaller sigma to exploit local structure quickly,
        # or larger to explore. Given the hidden shift/rotation, exploration is key initially.
        
        # Strategy: 
        # 1. Start with a few random initial points (Multi-Start).
        # 2. Run CMA-ES from the best one, or run parallel CMA-ES? 
        # Parallel CMA-ES is expensive. Sequential with restarts is safer for time.
        # Let's do: 
        # - Phase 1: Quick random search to find a promising region (50 calls).
        # - Phase 2: CMA-ES starting from the best random point.
        # - If CMA-ES stagnates (no improvement for X steps), restart CMA-ES from a new random point 
        #   or perturb the best point, keeping the global best.
        
        # Since Katsuura is highly multimodal, finding the global minimum (or close to it) 
        # might require exploring different basins.
        
        # Let's implement a simple Multi-Start CMA-ES.
        # We will run CMA-ES instances sequentially.
        # Each instance gets a sub-budget.
        # If an instance finds a new global best, we might restart CMA-ES from there with a smaller sigma.
        
        # Actually, a simpler and often effective strategy for BBOB f23 is:
        # 1. Use a population-based method that maintains diversity.
        # 2. CMA-ES with a restart rule: if the best function value doesn't improve for 
        #    a certain number of generations, restart CMA-ES with a new initial point 
        #    (possibly perturbed from the current best) and a fresh covariance matrix.
        
        # Let's implement CMA-ES from scratch (lightweight version) to avoid scipy overhead 
        # if scipy is slow to import or use, though scipy is available. 
        # Writing a basic CMA-ES is safer for control.
        
        # However, implementing full CMA-ES correctly is tricky. 
        # Let's use a simplified CMA-ES or a well-tuned (1+1)-ES with restarts?
        # The prompt inspiration 2 mentions "Multi-Start CMA-ES with adaptive restarts" scoring 1.2979.
        # Let's try to emulate that.
        
        # To keep it simple and robust within the time limit:
        # We will use a population-based search with a simple covariance adaptation 
        # (like a simplified CMA-ES) or just a robust (mu+lambda)-ES.
        # Given the 500 budget, a (10+10)-ES or similar might be good.
        
        # Let's stick to a robust (1+1)-ES with RESTARTS.
        # Why? (1+1)-ES is very fast per step. We can do many restarts.
        # Restart condition: If no improvement for `stagnation_threshold` steps, 
        # restart from a new random point or a point perturbed from the global best.
        
        # But Katsuura has many local minima. (1+1)-ES gets stuck easily.
        # Multi-start is key.
        
        # Strategy:
        # 1. Divide budget into chunks.
        # 2. For each chunk, run a local optimizer (e.g., (1+1)-ES) starting from a random point.
        # 3. Keep the best point found so far.
        # 4. Use the time guard to stop if needed.
        
        # Let's refine:
        # Phase 1: Random Search (50 evaluations) to find a good starting point.
        # Phase 2: Run (1+1)-ES from the best point found in Phase 1.
        # Phase 3: If (1+1)-ES stagnates, restart (1+1)-ES from a new random point 
        #          (or a point in the neighborhood of the global best).
        
        # Actually, the inspiration "Multi-Start CMA-ES" suggests CMA-ES is better.
        # Let's implement a basic CMA-ES.
        
        def cma_es(func, x0, sigma0, budget, rng, deadline, global_best_f, global_best_x):
            """
            Runs CMA-ES.
            Returns best_x, best_f, calls_used, global_best_f, global_best_x
            """
            n = len(x0)
            # CMA-ES parameters
            lambda_pop = int(4 + 3 * np.log(n))
            mu = lambda_pop // 2  # Number of recombination weights
            # Weights: proportional to log ranks
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1 / np.sum(w**2)
            
            # Step size adaptation
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initial state
            x = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = 1.0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            
            best_x = x.copy()
            best_f = func(x)
            calls = 1
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = x.copy()
            
            stagnation_counter = 0
            no_improve_threshold = 50  # Restart if no improvement for 50 generations
            
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                ys = []
                for i in range(lambda_pop):
                    # Sample from N(x, sigma^2 * B C B^T)
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    ys.append(y)
                
                # Evaluate population
                ys_f = []
                for y in ys:
                    fy = func(y)
                    calls += 1
                    ys_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        if best_f < global_best_f:
                            global_best_f = best_f
                            global_best_x = best_x.copy()
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    else:
                        stagnation_counter += 1
                
                # Sort by function value
                idx = np.argsort(ys_f)
                ys_sorted = [ys[i] for i in idx]
                ys_f_sorted = [ys_f[i] for i in idx]
                
                # Recombination
                x_new = sum(w[i] * ys_sorted[i] for i in range(mu))
                
                # Update covariance matrix
                # Calculate step size
                y_vec = (x_new - x) / sigma
                h_sigma = (np.sqrt((1 - np.exp(-2 / c_sigma)) * (1 - np.exp(-4 / c_sigma))) / (np.sqrt(c_sigma) * sigma)) * np.sqrt(np.trace(C) / n - 1)
                
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_vec
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update p_c
                h_1 = (1 - np.exp(-2 / c_c)) * (1 + (1 - np.exp(-4 / c_c))) / (np.sqrt(c_c) * sigma)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_vec
                
                # Update C
                delta_diag = h_1 * (1 - h_1) * np.outer(p_c, p_c)
                C = (1 + c_c * (c_1 + c_mu - c_c) - c_1 - c_mu) * C
                C += c_1 * (np.outer(y_vec, y_vec) - C)
                C += c_mu * sum(w[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu))
                
                # Ensure C is symmetric positive definite (SPD)
                C = (C + C.T) / 2
                # Eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                B = eigvecs
                D = np.sqrt(eigvals)
                
                # Update mean
                x = x_new
                
                # Check for stagnation
                if stagnation_counter > no_improve_threshold:
                    break
            
            return best_x, best_f, calls, global_best_f, global_best_x

        # Main loop
        # Phase 1: Random Search
        rng = np.random.default_rng(0)
        random_budget = 50
        x_best_random = None
        f_best_random = float('inf')
        
        for i in range(random_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < f_best_random:
                f_best_random = fx
                x_best_random = x.copy()
        
        global_best_x = x_best_random
        global_best_f = f_best_random
        calls_used = random_budget
        
        # Phase 2: CMA-ES with Restarts
        # We will run CMA-ES. If it stagnates, we restart.
        # How to restart? Pick a new random point or perturb the global best.
        # Let's try restarting from a new random point if the current CMA-ES doesn't improve 
        # the global best significantly.
        
        # To keep it simple, let's run one long CMA-ES but with a restart mechanism inside?
        # The function above breaks on stagnation. We can call it again.
        
        # Let's manage the budget carefully.
        # We have budget - calls_used left.
        # We will run CMA-ES instances.
        
        # Initial sigma for CMA-ES: 
        # If we start from a random point, a larger sigma might help explore.
        # But Katsuura is rugged. A smaller sigma might be better to converge to a local min.
        # Let's start with sigma=1.0.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Determine initial point for this CMA-ES instance
            # 50% chance to start from global best (exploit), 50% random (explore)
            if rng.random() < 0.5 and global_best_x is not None:
                x0 = global_best_x + 0.5 * rng.standard_normal(n)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.5
            else:
                x0 = rng.uniform(-5, 5, n)
                sigma0 = 1.0
            
            # Give this instance a chunk of the remaining budget
            # Or just let it run until it stagnates or budget is exhausted.
            # Let's give it a sub-budget of min(100, remaining) to force restarts.
            sub_budget = min(100, budget - calls_used)
            
            best_x, best_f, calls_in_instance, global_best_f, global_best_x = cma_es(
                func, x0, sigma0, calls_used + sub_budget, rng, deadline, 
                global_best_f, global_best_x
            )
            
            calls_used += calls_in_instance
            
            # If no improvement in global best, we might want to try a different strategy
            # But for now, just loop.
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
