"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        
        # DE parameters
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.5  # Crossover rate
        
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, n))
        fvals = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fvals)
        x_best_global = population[best_idx].copy()
        f_best_global = fvals[best_idx]
        
        # Adaptive parameters
        current_F = F
        current_CR = CR
        
        while eval_count < total_budget:
            # Generate offspring
            for i in range(pop_size):
                if eval_count >= total_budget:
                    break
                    
                # Select three distinct random indices
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len(set([r1, r2, r3, i])) == 4:
                        break
                
                # Differential mutation
                mutant = population[r1] + current_F * (population[r2] - population[r3])
                mutant = np.clip(mutant, -5, 5)
                
                # Binary crossover
                trial = population[i].copy()
                cross_mask = rng.random(n) < current_CR
                if not np.any(cross_mask):
                    cross_mask[rng.integers(n)] = True
                trial[cross_mask] = mutant[cross_mask]
                
                # Evaluate trial
                f_trial = func(trial)
                eval_count += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    population[i] = trial
                    fvals[i] = f_trial
                    if f_trial < f_best_global:
                        f_best_global = f_trial
                        x_best_global = trial.copy()
            
            # Adaptive parameter control
            # Reduce F and CR slightly over time to refine search
            progress = eval_count / total_budget
            current_F = F * (1.0 - 0.3 * progress)
            current_CR = CR * (1.0 - 0.2 * progress)
            
            # If no improvement in last generation, perturb the population
            # Simple strategy: occasionally inject random individuals
            if eval_count < total_budget and rng.random() < 0.1:
                # Replace worst individual with random one
                worst_idx = np.argmax(fvals)
                population[worst_idx] = rng.uniform(-5, 5, n)
                fvals[worst_idx] = func(population[worst_idx])
                eval_count += 1
                if fvals[worst_idx] < f_best_global:
                    f_best_global = fvals[worst_idx]
                    x_best_global = population[worst_idx].copy()
        
        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
