from scipy.optimize import minimize
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.func = func
        self.deadline = self.start_time + 280.0  # 20s safety margin from 300s limit
        self.best_x = None
        self.best_f = float('inf')
        
        # Track best seen for early termination of CMA-ES if it stalls
        self.last_improvement_call = 0
        self.last_best_f = float('inf')

        def wrapped_func(x):
            if self.call_count >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            x = np.asarray(x)
            if x.shape != (self.dim,):
                x = np.reshape(x, (self.dim,))
            
            self.call_count += 1
            f_val = float(func(x))
            
            if np.isfinite(f_val) and f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
                self.last_improvement_call = self.call_count
                self.last_best_f = f_val
            
            return f_val

        # Strategy: Multi-Start CMA-ES
        # CMA-ES is robust to rotation and scaling, which are key features of BBOB f21.
        # We split the budget into several restarts.
        # Each restart uses a Latin Hypercube sampling for the initial population to ensure coverage.
        
        # Heuristic: 
        # If dim is small (5), CMA-ES is very fast per generation.
        # We can afford many restarts.
        # Let's try 10 restarts.
        # Pop size for CMA-ES: lambda = 4 + floor(3 * ln(n)) -> 4 + 3*1.6 = 8.8 -> 9.
        # Let's use lambda = 12 for better exploration in early stages.
        
        lambda_pop = 12
        num_restarts = 10
        
        # Total budget allocation
        # Reserve some budget for potential final polish if needed, but CMA-ES handles polish well.
        # We will allocate budget dynamically.
        
        # Simple allocation: budget / num_restarts
        # But ensure enough for at least 2-3 generations per restart.
        # Min calls per restart: lambda_pop * 3 (initial + 2 gens)
        min_calls_per_restart = lambda_pop * 3
        max_restarts = max(1, self.budget // min_calls_per_restart)
        num_restarts = min(num_restarts, max_restarts)
        
        budget_per_restart = self.budget // num_restarts
        
        rng = np.random.default_rng(42)
        
        for r in range(num_restarts):
            if self.call_count >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            # Initialize CMA-ES state
            # x0: Random point in [-5, 5]^d
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Initial sigma: Scale to cover a significant part of the space
            # Standard deviation of uniform[-5,5] is ~2.88. 
            # Let's start with sigma = 1.5 to allow local search but not too global.
            # Or maybe 2.0?
            sigma = 2.0
            
            # CMA-ES Parameters
            # lambda = 12
            # mu = lambda // 2 = 6 (top 50% survive)
            # Weights: Standard CMA-ES weights
            # c_sigma, c_c, c_1, c_mu are standard.
            
            # We implement a simplified, robust CMA-ES loop here to avoid dependency on specific scipy.optimize.cmaes internals 
            # which might vary or be slow to import/init, and to have tight control over budget.
            
            # State
            x_m = x0.copy()
            B = np.eye(self.dim)
            D = np.ones(self.dim)
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Parameters
            lambda_ = lambda_pop
            mu = lambda_ // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            
            # Initial Population
            # Latin Hypercube sampling for better initial spread
            def latin_hypercube_sampling(n_samples, n_dims, lower, upper):
                points = []
                for i in range(n_dims):
                    # Shuffle indices
                    perm = rng.permutation(n_samples)
                    # Create stratified points
                    points_i = (perm + rng.random(n_samples)) / n_samples
                    points.append(points_i)
                points = np.array(points).T
                # Scale to [lower, upper]
                points = points * (upper - lower) + lower
                return points
            
            # Generate initial population
            # We need lambda_ points.
            # If lambda_ > 1, we can use LHS.
            if lambda_ > 1:
                pop = latin_hypercube_sampling(lambda_, self.dim, -5.0, 5.0)
            else:
                pop = np.array([x0])
            
            # Evaluate initial population
            fitness = np.full(lambda_, float('inf'))
            for i in range(lambda_):
                if self.call_count >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                fitness[i] = wrapped_func(pop[i])
            
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]
            
            # Update mean
            x_m = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # CMA-ES Main Loop
            # We run until budget for this restart is exhausted or convergence
            while self.call_count < self.budget and time.time() < self.deadline:
                # Check for convergence
                # 1. Sigma too small
                if sigma < 1e-12:
                    break
                
                # 2. No improvement for many generations (stall detection)
                # If we haven't improved the global best for, say, 2 * lambda_ evaluations, 
                # this restart is likely stuck. 
                # However, CMA-ES adapts. Let's use a simpler heuristic:
                # If the best fitness of the population hasn't improved in last lambda_ evaluations, stop.
                # This requires tracking local best per restart.
                
                # 3. Ill-conditioning
                if D.max() / D.min() > 1e14:
                    break
                
                # Generate new population
                new_pop = np.zeros((lambda_, self.dim))
                for i in range(lambda_):
                    if self.call_count >= self.budget:
                        break
                    if time.time() > self.deadline:
                        break
                    # Sample
                    z = rng.standard_normal(self.dim)
                    # Transform
                    step = sigma * (B @ (D * z))
                    new_pop[i] = x_m + step
                    # Clip to bounds? 
                    # CMA-ES is usually run on unbounded spaces. 
                    # The func clips to [-5,5]. 
                    # If we clip x before calling func, we might get stuck on boundaries.
                    # It is generally better to let the optimizer think in unbounded space 
                    # and let the func clip. The gradient information at the boundary 
                    # is effectively zero/undefined, but CMA-ES will just stop moving that way.
                    
                    new_pop[i] = np.clip(new_pop[i], -5.0, 5.0) # Ensure we don't pass out of bounds to func, though func clips anyway.
                    
                    new_pop[i] = wrapped_func(new_pop[i])
                
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                
                # Evaluate new population
                new_fitness = np.array([wrapped_func(x) for x in new_pop])
                
                # Combine with old population? No, standard CMA-ES replaces.
                # But we already evaluated initial pop. 
                # In this loop, we generate new ones.
                
                # Sort
                idx = np.argsort(new_fitness)
                new_pop = new_pop[idx]
                new_fitness = new_fitness[idx]
                
                # Update mean
                x_m = np.sum(weights[:, None] * new_pop[:mu], axis=0)
                
                # Update step size sigma
                # Standard CMA-ES update
                # h_sigma = 1 if ||p_sigma||^2 / (1 - (1-c_sigma)^{2/gen}) < (1.4 + 2/(n+1)) * chi_n else 0
                # For simplicity and speed, we use a simplified update or the standard one.
                # Let's use the standard CMA-ES update for robustness.
                
                # Calculate evolution path
                # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(mu_eff) * c_sigma * sum(w_i * z_i / sigma)
                # This requires storing the z vectors. 
                # To save memory and complexity, let's use a slightly simplified version 
                # or just rely on the covariance matrix update which is more important.
                # Actually, without p_sigma, the step size adaptation is less robust.
                # Let's implement the full update.
                
                # We need the z vectors from the selected individuals.
                # We didn't store them. 
                # Let's re-calculate or store them. 
                # Storing z is cheap.
                
                # Let's restart the loop logic to store z.
                # This is getting complex to implement from scratch correctly in a short time.
                # 
                # Alternative: Use SciPy's CMA-ES if available? 
                # scipy.optimize.minimize doesn't have CMA-ES directly.
                # There is a separate package `cmaes` or `pycma`. 
                # The prompt says "Python standard library, NumPy and SciPy are available."
                # It does NOT explicitly list `cmaes` or `pycma`. 
                # So I must implement it or use something else.
                
                # Given the risk of implementing CMA-ES incorrectly, 
                # and the fact that the current score is 0.97 (very high),
                # maybe a different strategy is safer.
                
                # Let's look at the inspirations.
                # Inspiration 1 (1.27) uses "Initial program... SCiEL evolves".
                # The current program (0.97) uses DE + NM.
                # 
                # What if I use a **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** 
                # implemented via a simpler, more robust algorithm like **Particle Swarm Optimization (PSO)**?
                # PSO is simpler to implement correctly and handles multimodality well with inertia.
                
                # Let's switch to **Inertia-Weighted PSO** with **Local Search** (Nelder-Mead) on the best particle.
                # PSO:
                # - Initialize swarm with LHS.
                # - Update velocities: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                # - Update positions: x = x + v
                # - Clip to bounds.
                # - Keep personal best and global best.
                # - After PSO runs for a portion of the budget, run NM on the global best.
                
                # This is a very different family from DE.
                
                # Let's implement PSO.
                
                # Reset state for this restart
                num_particles = 20
                max_iter_pso = budget_per_restart // num_particles
                
                if max_iter_pso < 10:
                    max_iter_pso = 10
                    # Adjust budget usage
                    pass
                
                # Initialize PSO
                # x: (N, D), v: (N, D), pbest: (N, D), pbest_f: (N,)
                # gbest: (D,), gbest_f: scalar
                
                # LHS for initial positions
                x_swarm = latin_hypercube_sampling(num_particles, self.dim, -5.0, 5.0)
                v_swarm = np.zeros_like(x_swarm)
                
                pbest = x_swarm.copy()
                pbest_f = np.full(num_particles, float('inf'))
                
                # Evaluate initial swarm
                for i in range(num_particles):
                    if self.call_count >= self.budget:
                        break
                    if time.time() > self.deadline:
                        break
                    pbest_f[i] = wrapped_func(x_swarm[i])
                    pbest[i] = x_swarm[i]
                
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                
                # Initialize global best
                idx_best = np.argmin(pbest_f)
                gbest = pbest[idx_best].copy()
                gbest_f = pbest_f[idx_best]
                
                # PSO Parameters
                w = 0.7  # Inertia
                c1 = 1.5 # Cognitive
                c2 = 1.5 # Social
                
                # Run PSO
                for it in range(max_iter_pso):
                    if self.call_count >= self.budget:
                        break
                    if time.time() > self.deadline:
                        break
                    
                    # Update velocities
                    r1 = rng.random((num_particles, self.dim))
                    r2 = rng.random((num_particles, self.dim))
                    
                    v_swarm = w * v_swarm + c1 * r1 * (pbest - x_swarm) + c2 * r2 * (gbest[None, :] - x_swarm)
                    
                    # Update positions
                    x_swarm = x_swarm + v_swarm
                    x_swarm = np.clip(x_swarm, -5.0, 5.0)
                    
                    # Evaluate new positions
                    for i in range(num_particles):
                        if self.call_count >= self.budget:
                            break
                        if time.time() > self.deadline:
                            break
                        
                        f_val = wrapped_func(x_swarm[i])
                        
                        if f_val < pbest_f[i]:
                            pbest_f[i] = f_val
                            pbest[i] = x_swarm[i]
                            
                            if f_val < gbest_f:
                                gbest_f = f_val
                                gbest = x_swarm[i].copy()
                
                # End of PSO for this restart
                
                # Local Polish with Nelder-Mead on the best found in this restart
                # Use remaining budget for this restart or global remaining budget?
                # Let's use a fixed small budget for polish, e.g., 20 evaluations.
                # Or use all remaining budget if this is the last restart?
                # Better: Allocate 10% of budget_per_restart for polish.
                
                polish_budget = max(10, budget_per_restart // 10)
                remaining_global = self.budget - self.call_count
                
                # Cap polish budget by what's left
                polish_budget = min(polish_budget, remaining_global)
                
                if polish_budget > 5 and gbest_f < float('inf'):
                    # Implement simple Nelder-Mead manually to avoid scipy dependency issues?
                    # Or just use scipy if it's fast enough.
                    # Since we are in a loop, importing scipy inside is bad.
                    # Let's import at top.
                    
                    # We already have `from scipy.optimize import minimize` in the original, 
                    # but I removed it in the draft. Let's add it back.
                    # Actually, let's stick to the plan:
                    # I will use `scipy.optimize.minimize` with Nelder-Mead.
                    
                    # We need to track calls carefully.
                    # The wrapped_func handles this.
                    
                    try:
                        # NM options
                        max_iter_nm = polish_budget // (self.dim + 1)
                        res = minimize(
                            wrapped_func, 
                            gbest, 
                            method='Nelder-Mead',
                            options={
                                'maxiter': max_iter_nm,
                                'xatol': 1e-8,
                                'fatol': 1e-8
                            }
                        )
                        if np.isfinite(res.fun) and res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = res.x
                    except Exception:
                        pass

        # Final Fallback if nothing found
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            if self.call_count < self.budget:
                self.best_f = wrapped_func(self.best_x)
            else:
                self.best_f = 0.0

        return self.best_x, self.best_f
