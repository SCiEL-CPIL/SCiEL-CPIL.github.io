import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 300.0  # Safety margin for 330s limit

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For n=5, lambda=10 is a good choice
        lam = 10
        mu = lam // 2
        
        # Weights for the recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        # Start at origin, which is a reasonable guess for the center of the box
        # If the shift is large, CMA-ES will move.
        mean = np.zeros(n)
        sigma = 2.0  # Initial standard deviation, covers [-4, 4] reasonably
        
        # State variables for CMA-ES
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Precompute Cholesky decomposition update helper
        def update_chol(C, step):
            # Simple Cholesky update is complex, we will recompute or use a stable method
            # For n=5, recomputing Cholesky is cheap
            pass
            
        def chol_update(C, B, D):
            # This is a simplified update. For robustness, we will just recompute
            # Cholesky of C at the end of each generation if needed, or use the
            # standard CMA-ES update rules.
            pass

        # Actually, implementing full CMA-ES with covariance matrix updates 
        # from scratch is error-prone. 
        # Alternative: Use scipy.optimize.differential_evolution? 
        # No, DE is often slower to converge in precision.
        # Alternative: Use a simpler (1+1)-ES with covariance adaptation?
        # Or just use a well-tuned Random Restart Local Search?
        
        # Let's try a simpler, very robust approach: 
        # 1. Global search with a coarse grid or random sampling.
        # 2. Local search with Nelder-Mead or L-BFGS-B.
        
        # However, the current program uses L-BFGS-B and scores 1.5.
        # CMA-ES is significantly better for this specific benchmark (BBOB f14).
        # Let's implement a correct CMA-ES.
        
        # To avoid bugs in manual CMA-ES implementation, I will use a 
        # simplified version that updates the mean and sigma, and uses 
        # a diagonal covariance initially, or just stick to the 
        # "Evolution Strategy" with adaptation of sigma and mean.
        
        # Actually, let's use the `cma` package if available? 
        # The prompt says "Python standard library, NumPy and SciPy are available."
        # It does NOT mention the `cma` package. So I must implement it.
        
        # I will implement a basic CMA-ES loop.
        
        best_x = None
        best_f = float('inf')
        
        def wrapped_func(x):
            nonlocal self
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)
        
        # Generate initial population
        for gen in range(budget // lam + 1):
            if self.call_count >= budget:
                break
            if time.time() - self.start_time > self.time_limit - 5:
                break
                
            # Generate offspring
            offspring = []
            for i in range(lam):
                # Sample from multivariate normal
                # z ~ N(0, C)
                # We need C_sqrt. 
                # For simplicity and stability in a short script, 
                # I will use a diagonal covariance matrix adaptation (ES with sigma adaptation)
                # which is less powerful than full CMA-ES but much simpler and less bug-prone.
                # Actually, for n=5, full CMA-ES is not that hard.
                
                # Let's try a simpler "Adaptive Random Search" or "1+1-ES" with 
                # multiple restarts?
                
                # Let's stick to the L-BFGS-B approach but improve the initial points.
                # The weakness of the current program is likely the random initial points.
                # If we use a better global search to find good starting points for L-BFGS-B,
                # we can improve the score.
                
                # Idea: Use Differential Evolution (DE) for the first 300 calls to find 
                # a good basin, then L-BFGS-B for the remaining 200 calls.
                
                pass
            
            # Let's pivot to a Hybrid Strategy:
            # Phase 1: Differential Evolution (Global)
            # Phase 2: L-BFGS-B (Local Polish)
            
            # DE Parameters
            # Population size NP = 10
            # F = 0.8, CR = 0.9
            # This uses NP calls per generation.
            # 300 calls / 10 = 30 generations.
            
            pass
            
        # Implementation of Hybrid DE + L-BFGS-B
        
        from scipy.optimize import minimize
        
        # Phase 1: Differential Evolution
        NP = 10
        F = 0.8
        CR = 0.9
        de_budget = 300
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5.0, 5.0, size=(NP, n))
        
        # Evaluate initial population
        fitness = np.zeros(NP)
        for i in range(NP):
            if self.call_count >= de_budget:
                break
            fitness[i] = wrapped_func(pop[i])
        
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # DE Loop
        while self.call_count < de_budget:
            if time.time() - self.start_time > self.time_limit - 5:
                break
                
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(NP)
            
            for i in range(NP):
                if self.call_count >= de_budget:
                    break
                if time.time() - self.start_time > self.time_limit - 5:
                    break
                    
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # DE/Rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.zeros(n)
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Evaluate
                trial = np.clip(trial, -5.0, 5.0)
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Update population
            pop = new_pop
            fitness = new_fitness
            
            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_f:
                best_f = fitness[current_best_idx]
                best_x = pop[current_best_idx].copy()
                
        # Phase 2: Local Optimization with L-BFGS-B
        # Use the best point from DE as the start.
        # Also, maybe try a few other top points from DE.
        
        remaining_budget = budget - self.call_count
        if remaining_budget > 10:
            # Try L-BFGS-B on the best point
            try:
                res = minimize(wrapped_func, best_x, method='L-BFGS-B', 
                               bounds=[(-5.0, 5.0) for _ in range(n)],
                               options={'maxfun': remaining_budget, 'maxiter': 100})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
