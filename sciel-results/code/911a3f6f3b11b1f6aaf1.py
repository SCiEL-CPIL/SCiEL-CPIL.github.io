"""New optimizer using Differential Evolution for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution based optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # DE Parameters
        # Population size
        npop = 10
        # Mutation factor
        F = 0.8
        # Crossover rate
        CR = 0.9
        # Strategy: DE/rand/1/bin
        
        # Bounds
        lb = -5.0
        ub = 5.0
        
        # Initialize population
        pop = np.random.uniform(lb, ub, size=(npop, n))
        fvals = np.zeros(npop)
        calls = 0
        
        # Evaluate initial population
        for i in range(npop):
            if calls >= budget:
                break
            fvals[i] = func(pop[i])
            calls += 1
            
        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Main DE loop
        while calls + npop <= budget:
            # Generate offspring
            for i in range(npop):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(npop, 3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, lb, ub)
                
                # Crossover: binomial crossover
                jrand = rng.integers(0, n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    
                # Evaluate trial
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        # If we have remaining budget, use it for a simple local search or just stop
        # Given the "new approach" constraint, I will stick to DE.
        # However, if calls < budget, we can do a few more random perturbations around best_x?
        # No, let's keep it pure DE. The loop condition `calls + npop <= budget` ensures we don't exceed.
        # If we have < npop calls left, we can't do a full generation.
        # We can do a partial generation or single-point mutation.
        
        # Let's add a "polishing" phase using simple random restarts or local jitter if budget remains
        while calls < budget:
            # Jitter around best
            scale = 0.5
            trial = best_x + rng.normal(0, scale, size=n)
            trial = np.clip(trial, lb, ub)
            f_trial = func(trial)
            calls += 1
            if f_trial < best_f:
                best_f = f_trial
                best_x = trial.copy()
            # Decay scale
            scale *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
