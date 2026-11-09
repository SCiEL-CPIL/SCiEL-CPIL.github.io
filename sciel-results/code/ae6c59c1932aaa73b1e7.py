"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """AHA (Adaptive Hyperparameter Annealing) optimizer for f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        x_global_best = np.random.uniform(-5, 5, dim)
        f_global_best = np.inf
        calls = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # AHA Strategy:
        # Single continuous search with a population of 10.
        # Instead of restarting, we anneal the step size (sigma) and
        # covariance adaptation based on the improvement rate.
        
        pop_size = 10
        # Initialize population around the global best (which is random initially)
        # We do a small random initialization to spread out
        population = np.random.uniform(-5, 5, (pop_size, dim))
        f_values = np.array([evaluate(x) for x in population])
        
        # Initial sigma: scale to box size
        sigma = 2.0
        # Covariance matrix (identity start)
        C = np.eye(dim)
        
        # Annealing parameters
        # We track the best improvement in the last K generations
        history_improvements = []
        window_size = 20
        
        # AHA specific:
        # sigma is multiplied by a factor that depends on the recent success rate.
        # If we are improving, we might keep sigma or increase it slightly to explore.
        # If we are stagnating, we decrease sigma to exploit.
        
        # To handle the multimodal nature of f22 without restarts, we rely on
        # the population maintaining diversity. If the population converges too much,
        # we might need to inject diversity. 
        # However, the prompt asks for "Adaptive Hyperparameter Annealing... 
        # where mutation step size and covariance adaptation rates are annealed 
        # based on the current improvement rate".
        
        # Let's define the adaptation:
        # success_rate = fraction of new points that beat the current best
        # if success_rate is high, sigma *= 1.1 (explore more)
        # if success_rate is low, sigma *= 0.9 (exploit more)
        
        # We also adapt the covariance matrix C based on the successful steps.
        
        while check_time() and calls < budget:
            # 1. Sort population by fitness
            idx = np.argsort(f_values)
            population = population[idx]
            f_values = f_values[idx]
            
            # 2. Update global best if needed (evaluate already does this, but keep local view)
            current_best_x = population[0]
            current_best_f = f_values[0]
            
            # 3. Calculate improvement rate for annealing
            # Compare best of this generation vs best of previous? 
            # Simpler: track if the best value improved in the last few steps.
            # For AHA, we often look at the relative change.
            
            # Generate new candidates
            new_population = []
            new_f_values = []
            
            # We will generate pop_size new candidates.
            # To maintain diversity, we don't just mutate the best.
            # We mutate a mix. Let's say top 50% are parents for the next gen?
            # Or just mutate all?
            # Standard CMA-ES mutates the best lambda/2.
            # Let's use the best 5 as parents to create 10 children.
            
            parents = population[:5]
            
            for i in range(pop_size):
                # Pick a random parent from the top 5
                p_idx = rng.integers(0, 5)
                x_parent = parents[p_idx]
                
                # Sample step
                z = rng.multivariate_normal(np.zeros(dim), C)
                x_child = x_parent + sigma * z
                x_child = np.clip(x_child, -5, 5)
                
                new_population.append(x_child)
            
            new_population = np.array(new_population)
            
            # Evaluate new population
            for x in new_population:
                if not check_time() or calls >= budget:
                    break
                f = evaluate(x)
                new_f_values.append(f)
            
            if len(new_f_values) < pop_size:
                # If we ran out of budget or time, pad with current best to maintain shape
                # This shouldn't happen in the loop condition, but just in case
                while len(new_f_values) < pop_size:
                    new_f_values.append(f_global_best)
                    new_population = np.vstack([new_population, [x_global_best]])
                break
            
            new_f_values = np.array(new_f_values)
            
            # 4. Merge and select
            # Combine old and new? Or just replace?
            # AHA usually replaces or selects from combined.
            # Let's select the best 10 from the combined set of 20 (old + new).
            # This preserves the best individuals (elitism) which helps with stagnation.
            
            combined_pop = np.vstack([population, new_population])
            combined_f = np.concatenate([f_values, new_f_values])
            
            idx = np.argsort(combined_f)
            population = combined_pop[idx[:pop_size]]
            f_values = combined_f[idx[:pop_size]]
            
            # 5. Adapt Hyperparameters (sigma and C)
            
            # Calculate success rate: how many of the NEW children were better than the 
            # previous best? Or just how many made it into the top 10?
            # Let's use a simpler metric: Did the global best improve in this generation?
            # If yes, success = 1.0, else 0.0? That's too binary.
            
            # Better metric: The fraction of new children that are better than the 
            # median of the previous population.
            prev_median = np.median(f_values) # This is current best 10 median
            # Actually, let's look at the improvement of the global best over the last 'window' steps.
            
            # For AHA, let's define "improvement rate" as the change in the best fitness 
            # normalized by the current fitness scale.
            
            # Simple adaptive rule:
            # If the best fitness improved in this generation (compared to the best of the 
            # previous generation's population), increase sigma slightly.
            # Otherwise, decrease it.
            
            # We need to store the previous best.
            if not hasattr(self, '_prev_best_f'):
                self._prev_best_f = np.inf
            
            prev_best = self._prev_best_f
            curr_best = f_values[0]
            
            # Normalize improvement
            # If curr_best < prev_best, we improved.
            if curr_best < prev_best:
                improvement = True
            else:
                improvement = False
            
            self._prev_best_f = curr_best
            
            # Anneal Sigma
            if improvement:
                sigma *= 1.05 # Expand slightly to explore new basin
            else:
                sigma *= 0.95 # Contract to exploit current basin
            
            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Adapt Covariance Matrix C
            # If we improved, we might want to align C with the direction of improvement.
            # If we didn't, we might want to keep C stable or slightly reset it.
            
            # A simple CMA-like adaptation:
            # If improvement, update C towards the outer product of the successful step.
            # We don't easily track the specific successful step in the merged population.
            
            # Alternative: Adapt C based on the spread of the current population.
            # If the population is diverse, keep C wide. If it converges, C should narrow.
            
            # Let's compute the covariance of the current population (centered at best).
            # This is expensive for dim=5? No, 5x5 is tiny.
            # But doing it every generation might be noisy.
            
            # Let's use a simpler heuristic for C:
            # If improvement, keep C as is or slightly increase its eigenvalues (explore).
            # If no improvement, slightly shrink eigenvalues (exploit).
            
            # To avoid full eigen-decomposition every step (though cheap for dim=5),
            # we can just scale C.
            # C = C * scale_factor
            
            if improvement:
                C = C * 1.01
            else:
                C = C * 0.99
            
            # Regularize C to prevent it from becoming singular or ill-conditioned
            # Add small identity
            C = 0.99 * C + 0.01 * np.eye(dim)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
