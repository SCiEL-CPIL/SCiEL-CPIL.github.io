"""Differential Evolution optimizer with adaptive F and CR parameters."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Population size: larger for better exploration
        pop_size = max(10, 5 * d)
        
        # Initialize population uniformly in [-5, 5]^d
        pop = rng.uniform(-5, 5, size=(pop_size, d))
        fitness = np.array([func(x) for x in pop])
        calls_made = pop_size
        
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Adaptive parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Track progress for adaptation
        no_improvement = 0
        best_f_history = [best_f]
        
        # Main loop
        while calls_made < budget:
            # Update best from current population
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_f:
                best_f = fitness[curr_best_idx]
                best_x = pop[curr_best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Adapt F and CR based on progress
            # If stuck, increase exploration
            if no_improvement > 10:
                F = min(1.0, F * 1.1)
                CR = min(1.0, CR * 1.05)
                no_improvement = 0
            else:
                # Gradually decrease F for exploitation
                F = max(0.3, F * 0.99)
                CR = max(0.5, CR * 0.999)
            
            # Generate offspring
            new_pop = pop.copy()
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # DE/rand/1 mutation strategy
                # Select three distinct indices different from i
                other_indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(other_indices, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)
                
                # Crossover
                r = rng.random(d) < CR
                if not r.any():
                    r[rng.integers(d)] = True  # Ensure at least one gene is crossed
                trial = np.where(r, v, pop[i])
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                ft = func(trial)
                calls_made += 1
                
                # Selection: keep the better one
                if ft <= fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        no_improvement = 0
            
            pop = new_pop
            
            # Check for convergence
            if np.std(fitness) < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
