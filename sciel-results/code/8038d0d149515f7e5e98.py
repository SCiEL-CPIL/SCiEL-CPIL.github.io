"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive Hyperparameter Annealing (AHA) with a single population.
    
    Uses a population of 10 candidates with adaptive step sizes and
    covariance adaptation based on improvement rates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 10
        
        # Initialize population with diverse starting points
        rng = np.random.default_rng(42)
        
        # Start with one good point and others spread out
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        
        # Create population around best point
        population = [x_best.copy()]
        f_values = [f_best]
        
        for i in range(1, pop_size):
            x_new = rng.uniform(-5, 5, dim)
            f_new = func(x_new)
            population.append(x_new)
            f_values.append(f_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
        
        # Track improvement rate for adaptive annealing
        recent_improvements = []
        max_improvements = 20
        
        # Adaptive parameters
        base_sigma = 2.0
        min_sigma = 0.005
        max_sigma = 6.0
        
        # Per-individual step sizes
        sigmas = [base_sigma] * pop_size
        
        # Crossover/mutation rates
        crossover_rate = 0.6
        mutation_rate = 0.4
        
        calls_made = len(f_values)
        
        for gen in range(budget - calls_made):
            if calls_made >= budget:
                break
            
            # Sort population by fitness
            indices = np.argsort(f_values)
            population = [population[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            sigmas = [sigmas[i] for i in indices]
            
            # Update best
            if f_values[0] < f_best:
                f_best = f_values[0]
                x_best = population[0].copy()
                recent_improvements.append(True)
            else:
                recent_improvements.append(False)
            
            # Keep only last N improvement records
            if len(recent_improvements) > max_improvements:
                recent_improvements = recent_improvements[-max_improvements:]
            
            # Calculate improvement rate
            if recent_improvements:
                improvement_rate = sum(recent_improvements) / len(recent_improvements)
            else:
                improvement_rate = 0.5
            
            # Adaptive annealing: adjust base sigma based on improvement rate
            # If improving well, increase exploration; if not, decrease
            if improvement_rate > 0.4:
                base_sigma *= 1.15
            elif improvement_rate < 0.15:
                base_sigma *= 0.85
            else:
                base_sigma *= 0.98
            
            base_sigma = np.clip(base_sigma, min_sigma, max_sigma)
            
            # Generate new population
            new_population = []
            new_f_values = []
            new_sigmas = []
            
            # Elitism: keep the best individual
            new_population.append(population[0].copy())
            new_f_values.append(f_values[0])
            new_sigmas.append(sigmas[0])
            
            for i in range(1, pop_size):
                # Choose parents: tournament selection
                # Pick two random individuals from top half
                parent1_idx = rng.choice(min(len(population), max(1, len(population) // 2)))
                parent2_idx = rng.choice(min(len(population), max(1, len(population) // 2)))
                
                p1 = population[parent1_idx]
                p2 = population[parent2_idx]
                
                # Crossover with probability crossover_rate
                if rng.random() < crossover_rate:
                    # Blended crossover
                    alpha = rng.random()
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()
                
                # Mutation
                if rng.random() < mutation_rate:
                    # Adaptive sigma for this individual
                    sigma_i = sigmas[i] if i < len(sigmas) else base_sigma
                    # Scale sigma based on improvement rate
                    sigma_i = sigma_i * (0.3 + improvement_rate)
                    sigma_i = np.clip(sigma_i, min_sigma, max_sigma)
                    
                    child = child + sigma_i * rng.standard_normal(dim)
                    
                    # Update sigma based on success
                    if rng.random() < 0.4:
                        sigmas[i] = sigma_i * 1.3
                    else:
                        sigmas[i] = sigma_i * 0.75
                    sigmas[i] = np.clip(sigmas[i], min_sigma, max_sigma)
                else:
                    sigma_i = sigmas[i] if i < len(sigmas) else base_sigma
                    child = child + sigma_i * 0.4 * rng.standard_normal(dim)
                
                # Clip to bounds
                child = np.clip(child, -5, 5)
                
                new_population.append(child)
                new_sigmas.append(sigmas[i] if i < len(sigmas) else base_sigma)
            
            # Evaluate new individuals (excluding the elite which is already evaluated)
            for i in range(1, pop_size):
                if calls_made >= budget:
                    break
                f_val = func(new_population[i])
                new_f_values.append(f_val)
                calls_made += 1
                
                if f_val < f_best:
                    f_best = f_val
                    x_best = new_population[i].copy()
            
            # If we ran out of budget, pad with the best value
            while len(new_f_values) < pop_size:
                new_f_values.append(f_best)
            
            population = new_population
            f_values = new_f_values
            sigmas = new_sigmas
        
        return x_best, f_best
# EVOLVE-BLOCK-END
