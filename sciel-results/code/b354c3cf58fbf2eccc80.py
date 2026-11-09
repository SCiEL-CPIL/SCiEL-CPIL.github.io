import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, return before 330s wall clock kill
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin
        
        rng = np.random.default_rng()
        
        best_x = None
        best_f = float('inf')
        
        calls_used = 0
        max_calls = self.budget
        
        # Strategy: Population-based (1+lambda) Evolutionary Strategy
        # Unlike the current (1+1) RHC which only perturbs the single best point,
        # we maintain a population of 'mu' best individuals.
        # This allows us to explore multiple basins simultaneously and avoids
        # premature convergence to a single local minimum more effectively
        # than a single greedy trajectory.
        
        # Parameters
        mu = 5  # Number of best individuals to keep (parents)
        lam = 10  # Number of offspring to generate per generation
        # Total population size is mu + lam = 15
        # With 500 budget, we can do ~33 generations.
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(mu, self.dim))
        evals = np.empty(mu)
        
        # Evaluate initial population
        for i in range(mu):
            if time.time() > deadline:
                return best_x, best_f
            fx = func(pop[i])
            calls_used += 1
            evals[i] = fx
            if fx < best_f:
                best_f = fx
                best_x = pop[i].copy()
        
        # Sort initial population by fitness (ascending)
        # We keep the best 'mu'
        idx = np.argsort(evals)
        pop = pop[idx]
        evals = evals[idx]
        
        # Adaptive step size logic:
        # Start with a moderate step. If we see progress, keep it or shrink slightly.
        # If we stagnate, expand.
        # For simplicity and robustness in a short budget, we use a fixed step size 
        # scaled by the box size, or a simple adaptation.
        # Let's use a step size that adapts based on success rate.
        step_size = 1.5
        
        # Main loop
        while calls_used < max_calls and time.time() < deadline:
            # Check if we have enough budget for a full generation
            # If not, we might do partial generation or stop.
            # For simplicity, if remaining budget < lam, we just do individual steps
            # or stop if too little. Let's allow partial generation.
            
            offspring = np.empty((0, self.dim))
            off_evals = np.empty(0)
            
            # Generate offspring
            # We generate 'lam' offspring from the 'mu' parents.
            # Each parent produces roughly lam/mu offspring.
            # To maintain diversity, we can sample parents with replacement or 
            # assign specific counts. Simple approach: each of the 'mu' parents 
            # produces 'lam // mu' or 'lam // mu + 1' offspring.
            
            off_per_parent = lam // mu
            extra = lam % mu
            
            current_off_idx = 0
            for i in range(mu):
                n_off = off_per_parent + (1 if i < extra else 0)
                
                for j in range(n_off):
                    if calls_used >= max_calls or time.time() > deadline:
                        break
                    
                    # Sample a parent (biased towards better ones? Or just use current i?)
                    # Standard (mu+lambda) or (mu,lambda). 
                    # Let's use (mu+lambda) selection: keep best of parents+offspring.
                    # But to avoid losing diversity completely, maybe (mu, lambda) is safer?
                    # (mu, lambda) discards parents. This encourages exploration.
                    # Let's try (mu+lambda) first as it's more standard for stability.
                    
                    # Perturb parent i
                    noise = rng.normal(0, step_size, self.dim)
                    x_new = pop[i] + noise
                    x_new = np.clip(x_new, -5, 5)
                    
                    fx_new = func(x_new)
                    calls_used += 1
                    
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                        
                    # Store offspring
                    if offspring.size == 0:
                        offspring = x_new.reshape(1, -1)
                        off_evals = np.array([fx_new])
                    else:
                        offspring = np.vstack([offspring, x_new])
                        off_evals = np.append(off_evals, fx_new)
                        
                if calls_used >= max_calls or time.time() > deadline:
                    break
                    
            if calls_used >= max_calls or time.time() > deadline:
                break
            
            # Combine parents and offspring
            # Parents: pop, evals
            # Offspring: offspring, off_evals
            
            all_pop = np.vstack([pop, offspring])
            all_evals = np.concatenate([evals, off_evals])
            
            # Select best 'mu'
            idx = np.argsort(all_evals)[:mu]
            pop = all_pop[idx]
            evals = all_evals[idx]
            
            # Adaptive Step Size
            # Simple heuristic: 
            # If best_f improved in this generation, shrink step slightly.
            # If no improvement, expand step slightly.
            # We need to track previous best.
            # Let's implement a simple adaptation.
            # Actually, with only 30 generations, fixed step might be fine.
            # But let's try a simple adaptation:
            # success = (min(all_evals) < min(evals)) # Did we find something better than best parent?
            # This is always true if we found a new global best.
            
            # Let's use a more robust adaptation:
            # Track the ratio of successful offspring (better than their parent).
            # For simplicity, let's just keep step_size constant at 1.5 or 2.0.
            # 1.5 is ~30% of the box width (10). This is reasonable for 5D.
            # If we stagnate, we might want to increase it.
            
            # Let's add a simple stagnation counter.
            # If no improvement in best_f for 5 generations, double step_size.
            # If improvement, reset counter and maybe shrink step_size by 0.9.
            
            # We need to store prev_best_f outside the loop.
            # Let's refactor slightly to include this logic properly.
            # Since I can't easily refactor the whole loop structure without risk, 
            # let's stick to a fixed step size for this "different strategy" implementation 
            # to ensure correctness, as the main difference is Population vs Single Trajectory.
            
            # Actually, let's add a simple step size adaptation based on global progress.
            # If best_f hasn't changed in the last 5 generations, increase step_size.
            # If it has, decrease slightly.
            
            # To do this, I need to track history.
            # Let's add a list to track best_f per generation.
            # But the loop above doesn't explicitly mark generation boundaries for history.
            
            # Let's keep it simple: Fixed step size. The diversity of the population 
            # is the key differentiator here.
            
            pass # End of generation
            
        return best_x, best_f
