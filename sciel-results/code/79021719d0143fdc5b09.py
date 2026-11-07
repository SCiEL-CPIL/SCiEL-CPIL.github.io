"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based Evolution Strategy with Stagnation Restarts.
    
    Strategy:
    1. Use a population of size lambda to explore the space.
    2. Adapt step size sigma using the 1/5 success rule.
    3. If no improvement is found for a certain number of generations (stagnation),
       restart the population centered around the global best with a larger sigma 
       to escape local minima (crucial for bi-modal functions like Lunacek).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin: 15s buffer before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # ES Parameters
        # For dim=5, lambda=10 is a good balance between diversity and speed
        lambda_ = 10
        mu = lambda_ // 2  # Select top half
        
        # Weights for recombination (logarithmic)
        weights = np.log(np.arange(mu, 0, -1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates for sigma adaptation
        # Standard 1/5 rule parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        
        # State variables
        global_best_x = None
        global_best_f = np.inf
        
        calls_used = 0
        rng = np.random.default_rng(42)
        
        # Stagnation detection
        stagnation_counter = 0
        stagnation_threshold = 15  # Generations without improvement to trigger restart
        
        # Main loop: We run generations until budget or time is exhausted
        # We implement a "soft restart" mechanism inside the loop
        
        # Initial population
        # Start with a reasonable spread. The box is [-5, 5].
        # Initial sigma should cover a significant portion of the box.
        sigma = 1.5 
        
        # Initialize population around origin or random?
        # Lunacek has a global minimum and a local minimum. 
        # Starting at 0 is often near the local minimum in standard instances, 
        # but the task says hidden shift/rotation. 
        # Let's start with a population spread over [-3, 3] to ensure we sample broadly.
        
        # Generate initial population
        pop = rng.uniform(-3, 3, size=(lambda_, dim))
        
        # Evaluate initial population
        f_pop = np.array([func(x) for x in pop])
        calls_used += lambda_
        
        # Update global best
        best_idx = np.argmin(f_pop)
        global_best_f = f_pop[best_idx]
        global_best_x = pop[best_idx].copy()
        
        # Sort population by fitness (ascending)
        order = np.argsort(f_pop)
        pop = pop[order]
        f_pop = f_pop[order]
        
        # Mean of selected individuals
        mean = np.average(pop[:mu], axis=0, weights=weights)
        
        gen = 0
        
        while calls_used < budget:
            # Time check
            if time.time() > deadline:
                break
                
            # Check for stagnation
            if stagnation_counter >= stagnation_threshold:
                # Restart: Re-center population around global best
                # Use a larger sigma to explore again
                # This helps escape the wrong basin in bi-Rastrigin
                sigma = max(sigma * 2.0, 1.0) 
                # Re-initialize population around global best
                # Keep some diversity
                pop = global_best_x[None, :] + sigma * rng.standard_normal((lambda_, dim))
                # Clip to bounds
                pop = np.clip(pop, -5, 5)
                
                # Evaluate new population
                # We need to be careful not to exceed budget
                # If we don't have enough budget for a full population, we break
                if calls_used + lambda_ > budget:
                    # Only evaluate as many as we can afford? 
                    # Actually, if we can't afford a full population, the ES logic breaks.
                    # Let's just break if we can't do a full generation.
                    break
                    
                f_pop = np.array([func(x) for x in pop])
                calls_used += lambda_
                
                # Update global best
                best_idx = np.argmin(f_pop)
                if f_pop[best_idx] < global_best_f:
                    global_best_f = f_pop[best_idx]
                    global_best_x = pop[best_idx].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1 # Still stagnant?
                    
                # Sort
                order = np.argsort(f_pop)
                pop = pop[order]
                f_pop = f_pop[order]
                
                # Recalculate mean
                mean = np.average(pop[:mu], axis=0, weights=weights)
                
                gen += 1
                continue
            
            # Standard ES Step
            
            # 1. Generate offspring
            # Sample from N(mean, sigma^2 * I)
            # We use the current mean (from previous selection)
            # Note: In standard (mu+lambda) or (mu,lambda), we generate new points.
            # Here we use (mu, lambda) strategy: replace all with new offspring.
            
            # Check if we have enough budget for a full population
            if calls_used + lambda_ > budget:
                # If we can't afford a full population, we might do a single step (1+1)
                # or just break. Breaking is safer to avoid partial updates that confuse the ES.
                # However, we can do a smaller step.
                # Let's just break to maintain strict ES consistency.
                break
                
            # Generate new population
            z = rng.standard_normal((lambda_, dim))
            new_pop = mean[None, :] + sigma * z
            new_pop = np.clip(new_pop, -5, 5)
            
            # Evaluate
            new_f = np.array([func(x) for x in new_pop])
            calls_used += lambda_
            
            # Combine with old population? No, (mu, lambda) replaces.
            # But for robustness against getting stuck, let's keep the global best 
            # in the pool for selection? 
            # Standard ES: select from new_pop only.
            # Let's stick to standard (mu, lambda) for simplicity and speed.
            
            # Update global best
            best_idx_new = np.argmin(new_f)
            if new_f[best_idx_new] < global_best_f:
                global_best_f = new_f[best_idx_new]
                global_best_x = new_pop[best_idx_new].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Sort new population
            order = np.argsort(new_f)
            new_pop = new_pop[order]
            new_f = new_f[order]
            
            # Select top mu
            selected = new_pop[:mu]
            
            # Calculate new mean
            mean = np.average(selected, axis=0, weights=weights)
            
            # Adapt sigma
            # Success rate: number of selected individuals that improved over the previous mean?
            # In (mu, lambda), the "success" is often defined by the improvement of the mean 
            # or the fraction of offspring better than the previous best.
            # Simpler 1/5 rule: Count how many of the new_pop are better than the previous mean?
            # Or better: Count how many of the top mu are better than the previous best?
            
            # Let's use the standard 1/5 rule on the success of the step.
            # Success if new mean is closer to global best? No, that's biased.
            # Standard: Success if the new point is better than the old point.
            # In population ES, a common heuristic is:
            # success_rate = (number of offspring better than previous best) / lambda
            # But previous best is global_best_f.
            
            # Let's use the improvement of the population mean.
            # If the new mean is "better" (lower f) than the old mean, it's a success.
            # We need f(mean_old). We didn't store it.
            
            # Alternative: Use the fraction of selected individuals that are better than the 
            # median of the population? 
            
            # Let's use a simpler heuristic for sigma adaptation in population ES:
            # If the best of the new population is better than the best of the old population, 
            # it's a "success" for the step. But we don't keep the old population in (mu, lambda).
            
            # Let's switch to a (mu+lambda) variant or just use a fixed sigma decay?
            # No, adaptive is better.
            
            # Let's define success as: The new global best is better than the old global best.
            # This is very sparse.
            
            # Better: Use the 1/5 rule on the *step* taken by the mean.
            # We need f(mean_old). Let's estimate it or store it.
            # Storing f(mean) costs 1 call per generation. With lambda=10, that's 10% overhead.
            # Budget 500 -> ~45 generations. 45 extra calls is significant.
            
            # Let's avoid evaluating the mean.
            # Instead, adapt sigma based on the variance of the selected population.
            # If variance is too high, decrease sigma. If too low, increase.
            
            # Variance of selected points
            if mu > 1:
                var = np.var(selected, axis=0).mean()
                # Target variance: Heuristic
                # If var is very small, we are converging, maybe too fast?
                # If var is large, we are exploring.
                
                # This is tricky without a target.
                
            # Let's stick to the 1/5 rule but approximate success.
            # Success if the best of the new population is better than the *second best* of the new population? No.
            
            # Let's use the "success" of the top individual relative to the previous best.
            # We track `global_best_f`.
            # If `new_f[0] < global_best_f` (before update), it was a success.
            # This is what we did above (stagnation_counter logic).
            # We can use the ratio of successful generations to adjust sigma.
            
            # Let's use a simple heuristic:
            # If we improved (stagnation_counter == 0), increase sigma slightly? No, usually decrease if we are converging?
            # Actually, if we are improving, we might want to keep exploring.
            # If we are NOT improving, we should explore more (increase sigma) or restart.
            
            # The restart mechanism handles the "no improvement" case by increasing sigma.
            # So for the normal step, we can keep sigma constant or slowly decay it?
            # If we keep sigma constant, we might oscillate.
            
            # Let's implement a simple sigma adaptation:
            # Keep a history of improvements.
            # If improved, sigma *= 1.1 (explore a bit more around the new best)
            # If not improved, sigma *= 0.9 (converge tighter)
            # This is a heuristic, but often works well for black-box.
            
            if stagnation_counter == 0:
                sigma *= 1.05 # Slight expansion to ensure we don't get stuck in a tiny local min
            else:
                sigma *= 0.95 # Slight contraction to refine
            
            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)
            
            gen += 1
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
