"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB-style Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        NP = 10  # Population size
        F = 0.8  # Mutation factor
        CR = 0.9 # Crossover rate
        
        rng = np.random.default_rng(42)
        
        # Handle very small budgets
        if budget < NP:
            best_x = np.zeros(dim)
            best_f = np.inf
            for _ in range(budget):
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            return best_x, best_f

        # Initialize population uniformly in [-5, 5]
        X = rng.uniform(-5, 5, (NP, dim))
        fvals = np.array([func(x) for x in X])
        
        best_idx = np.argmin(fvals)
        best_x = X[best_idx].copy()
        best_f = fvals[best_idx]
        
        calls_made = NP
        
        # Main loop
        while calls_made < budget:
            for i in range(NP):
                if calls_made >= budget:
                    break
                
                # Pick 3 distinct random indices
                a, b, c = rng.choice(NP, 3, replace=False)
                
                # Mutation: DE/rand/1
                v = X[a] + F * (X[b] - X[c])
                
                # Crossover: Binomial
                trial = X[i].copy()
                mask = rng.random(dim) < CR
                # Ensure at least one bit is taken from V
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                trial[mask] = v[mask]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls_made += 1
                
                if f_trial < fvals[i]:
                    X[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
