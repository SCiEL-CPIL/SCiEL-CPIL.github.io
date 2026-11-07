"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution based optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        # We need to track function calls to stay within budget
        # DE from scipy can be hard to control exactly, so we implement a custom DE
        # Or we can use scipy's with a wrapper that stops at budget
        
        # Since scipy's differential_evolution doesn't easily allow early stopping based on call count
        # in a way that guarantees exactly <= budget, let's implement a simple DE ourselves
        # or use a different approach.
        
        # Actually, let's implement a robust DE with careful budget management.
        # Weierstrass is multimodal and fractal, so DE is a good choice.
        
        # Initialize population
        npop = 10  # population size
        self._call_count = 0
        
        # Initial population: uniform in [-5, 5]
        population = np.random.uniform(-5, 5, (npop, self.dim))
        
        # Evaluate initial population
        fitness = np.zeros(npop)
        for i in range(npop):
            if self._call_count >= self.budget:
                break
            fitness[i] = func(population[i])
            self._call_count += 1
        
        # If we ran out of budget during initialization, return best
        if self._call_count >= self.budget:
            best_idx = np.argmin(fitness[:self._call_count])
            return population[best_idx], fitness[best_idx]
        
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        
        # Main loop
        while self._call_count < self.budget:
            best_idx = np.argmin(fitness)
            best_x = population[best_idx].copy()
            best_f = fitness[best_idx]
            
            for i in range(npop):
                if self._call_count >= self.budget:
                    break
                    
                # Select three distinct indices different from i
                r1, r2, r3 = i, i, i
                while r1 == i:
                    r1 = np.random.randint(npop)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(npop)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = np.random.randint(npop)
                
                # Create trial vector
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = population[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                self._call_count += 1
                
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
            
            # Check if we should stop (convergence)
            # Simple convergence check: if best hasn't improved in many generations
            # For now, just let it run until budget is exhausted
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
# EVOLVE-BLOCK-END
