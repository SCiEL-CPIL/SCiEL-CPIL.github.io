"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(mu+lambda)-ES with stagnation-aware step size adaptation for multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Parameters
        mu = 5
        lam = 10
        sigma = 2.0
        
        # Initialize population
        pop = rng.uniform(-5, 5, (mu, self.dim))
        pop_values = np.array([func(x) for x in pop])
        
        # Sort population by fitness
        idx = np.argsort(pop_values)
        pop = pop[idx]
        pop_values = pop_values[idx]
        
        best_val = pop_values[0]
        best_x = pop[0]
        
        calls_made = mu
        max_calls = self.budget
        
        while calls_made + lam <= max_calls:
            if time.time() >= deadline:
                break
                
            # Generate offspring
            offspring = []
            # Each parent generates lam/mu offspring
            offspring_per_parent = lam // mu
            for i in range(mu):
                for _ in range(offspring_per_parent):
                    y = np.clip(pop[i] + sigma * rng.standard_normal(self.dim), -5, 5)
                    offspring.append(y)
            
            # If lam is not divisible by mu, generate extra from random parents
            if len(offspring) < lam:
                for _ in range(lam - len(offspring)):
                    parent_idx = rng.integers(0, mu)
                    y = np.clip(pop[parent_idx] + sigma * rng.standard_normal(self.dim), -5, 5)
                    offspring.append(y)
            
            offspring = np.array(offspring)
            offspring_values = np.array([func(x) for x in offspring])
            calls_made += lam
            
            # Combine parents and offspring
            all_points = np.vstack((pop, offspring))
            all_values = np.concatenate((pop_values, offspring_values))
            
            # Select best mu
            idx = np.argsort(all_values)
            new_pop = all_points[idx[:mu]]
            new_pop_values = all_values[idx[:mu]]
            
            # Check for improvement
            new_best_val = new_pop_values[0]
            if new_best_val < best_val:
                best_val = new_best_val
                best_x = new_pop[0]
                # Improved: Refine slightly
                sigma *= 0.9
            else:
                # Stagnation: Expand to escape local minima
                sigma *= 1.2
            
            # Update population
            pop = new_pop
            pop_values = new_pop_values

        return best_x, best_val
# EVOLVE-BLOCK-END
