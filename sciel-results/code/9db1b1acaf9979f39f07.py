"""Rewritten optimizer using population-based differential evolution for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based differential evolution with adaptive parameters."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initial population: diverse points in [-5, 5]^dim
        # Use a mix of random and structured initializations
        pop_size = 20
        x0 = np.array([0.0] * dim)  # start near center
        # Generate initial population with good spread
        pop = np.empty((pop_size, dim))
        pop[0] = x0
        # Fill rest with random points, some closer to center
        for i in range(1, pop_size):
            if i % 3 == 0:
                pop[i] = np.random.uniform(-2, 2, dim)
            elif i % 3 == 1:
                pop[i] = np.random.uniform(-5, 5, dim)
            else:
                pop[i] = np.random.uniform(-3, 3, dim)
        
        # Evaluate initial population
        fx = np.empty(pop_size)
        for i in range(pop_size):
            fx[i] = func(pop[i])
        
        # Track best
        best_idx = np.argmin(fx)
        best_x = pop[best_idx].copy()
        best_f = fx[best_idx]
        
        calls_made = pop_size
        
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        
        # Adaptive strategy: use multiple mutation strategies
        # Strategy 0: DE/rand/1
        # Strategy 1: DE/best/1
        # Strategy 2: DE/rand-to-best/1
        
        stagnation_counter = 0
        no_improve = 0
        F_adapt = F
        CR_adapt = CR
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine mutation strategy based on progress
            if no_improve < 20:
                strategy = 1  # best/1 for exploitation
            elif no_improve < 50:
                strategy = 0  # rand/1 for exploration
            else:
                strategy = 2  # rand-to-best/1
            
            # Adaptive F and CR
            if no_improve > 30:
                F_adapt = min(1.0, F_adapt * 1.1)
                CR_adapt = min(0.95, CR_adapt + 0.05)
            elif no_improve < 10:
                F_adapt = max(0.3, F_adapt * 0.95)
                CR_adapt = max(0.7, CR_adapt - 0.02)
            
            # Generate offspring
            improved = False
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Select indices different from i
                available = [j for j in range(pop_size) if j != i]
                
                if strategy == 0:  # DE/rand/1
                    idxs = np.random.choice(available, 3, replace=False)
                    v = pop[idxs[0]] + F_adapt * (pop[idxs[1]] - pop[idxs[2]])
                elif strategy == 1:  # DE/best/1
                    idxs = np.random.choice(available, 2, replace=False)
                    v = best_x + F_adapt * (pop[idxs[0]] - pop[idxs[1]])
                else:  # DE/rand-to-best/1
                    idxs = np.random.choice(available, 3, replace=False)
                    v = pop[idxs[0]] + F_adapt * (best_x - pop[idxs[0]]) + F_adapt * (pop[idxs[1]] - pop[idxs[2]])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Binomial crossover
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR_adapt or j == jrand:
                        trial[j] = v[j]
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                if f_trial <= fx[i]:
                    pop[i] = trial
                    fx[i] = f_trial
                    improved = True
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if not improved:
                stagnation_counter += 1
                if stagnation_counter > 5:
                    # Restart some individuals with random jumps
                    for _ in range(3):
                        idx = np.random.randint(pop_size)
                        pop[idx] = np.random.uniform(-5, 5, dim)
                        fx[idx] = func(pop[idx])
                        calls_made += 1
                        if fx[idx] < best_f:
                            best_f = fx[idx]
                            best_x = pop[idx].copy()
                            no_improve = 0
                    stagnation_counter = 0
            
        return best_x, best_f
# EVOLVE-BLOCK-END
