"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE and (1+1)-ES optimizer for Büche-Rastrigin function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        global_best_x = np.zeros(dim)
        global_best_fx = func(global_best_x)
        calls_made = 1

        # Phase 1: Differential Evolution (Global Search)
        # Use 80% of budget
        de_budget = int(budget * 0.8)
        npop = 20
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(npop, dim))
        pop_fx = np.array([func(x) for x in pop])
        calls_made += npop
        
        # Update global best
        best_idx = np.argmin(pop_fx)
        if pop_fx[best_idx] < global_best_fx:
            global_best_fx = pop_fx[best_idx]
            global_best_x = pop[best_idx].copy()

        # DE Loop
        while calls_made < de_budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            for i in range(npop):
                if calls_made >= de_budget:
                    break
                
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(npop, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(npop, 3, replace=False)
                
                # Create trial vector
                trial = pop[i].copy()
                # DE/rand/1/bin
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                jrand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= pop_fx[i]:
                    pop[i] = trial
                    pop_fx[i] = f_trial
                    if f_trial < global_best_fx:
                        global_best_fx = f_trial
                        global_best_x = trial.copy()

        # Phase 2: (1+1)-ES Polish (Local Search)
        if calls_made < budget:
            sigma = 0.5
            x_cur = global_best_x.copy()
            f_cur = global_best_fx
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                y = x_cur + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    sigma *= 1.2
                    if fy < global_best_fx:
                        global_best_fx = fy
                        global_best_x = y
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
