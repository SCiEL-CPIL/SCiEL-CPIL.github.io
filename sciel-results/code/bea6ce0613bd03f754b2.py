"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Parameters
        NP = 20  # Population size
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Track evaluations
        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(self.dim)
        
        def count_func(x):
            nonlocal eval_count, best_f, best_x
            eval_count += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(NP, self.dim))
        pop_f = np.empty(NP)
        
        for i in range(NP):
            if eval_count >= self.budget:
                break
            pop_f[i] = count_func(pop[i])
            
        # If budget is too small for full population, return best
        if eval_count >= self.budget:
            return best_x, best_f
            
        # Differential Evolution loop
        # Each generation consumes NP evaluations
        while eval_count + NP <= self.budget:
            for i in range(NP):
                if eval_count >= self.budget:
                    break
                
                # Select 3 distinct random indices different from i
                while True:
                    a = rng.integers(NP)
                    b = rng.integers(NP)
                    c = rng.integers(NP)
                    if a != i and b != i and c != i and a != b and b != c and a != c:
                        break
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Boundary handling (clip to [-5, 5])
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                trial_f = count_func(trial)
                
                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

        # Use remaining budget for a few random restarts or local refinement?
        # DE is usually good enough, but if we have leftover budget (500 - 20 - 24*20 = 0), 
        # we might have used exactly 500.
        # 20 initial + 24 generations * 20 = 500. 
        # So we likely use exactly the budget.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
