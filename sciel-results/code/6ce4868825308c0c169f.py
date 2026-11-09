"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution based continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Use a fixed seed for reproducibility within the instance
        seed = 42
        
        # Define the bounds for the box [-5, 5]^dim
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Differential Evolution parameters:
        # - popsize: controls the population size (multiplier of dim)
        # - maxiter: we control iterations via budget, but DE handles internal steps.
        #   However, DE's callback or maxiter might not map 1:1 to function evaluations easily.
        #   Instead, we can use a simpler, more controllable algorithm like CMA-ES or a custom DE.
        
        # Let's implement a custom, lightweight Differential Evolution to strictly control the budget.
        # Standard DE:
        # 1. Initialize population of size NP.
        # 2. For each generation:
        #    a. Generate trial vectors using mutation (x_rand1 + F * (x_rand2 - x_rand3)).
        #    b. Evaluate trial vectors.
        #    c. Selection: compare trial vs. target, keep the better.
        
        # To fit budget=500, dim=5:
        # Let NP = 10 (small population for low dim).
        # Initial cost: 10 evaluations.
        # Remaining budget: 490.
        # Each generation costs NP evaluations.
        # Max generations: 490 / 10 = 49.
        
        rng = np.random.default_rng(seed)
        
        # Population size
        NP = 10
        # Mutation factor
        F = 0.8
        # Crossover rate
        CR = 0.9
        
        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5.0, 5.0, size=(NP, self.dim))
        fit = np.array([func(ind) for ind in pop])
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # Count evaluations used
        evals_used = NP
        remaining = self.budget - evals_used
        
        # Main loop
        while remaining > 0:
            # Determine how many individuals we can evaluate this generation
            # We must evaluate exactly NP individuals to maintain population structure,
            # unless remaining < NP. If remaining < NP, we evaluate remaining and stop.
            n_evals_this_gen = min(NP, remaining)
            
            # We need to be careful: if n_evals_this_gen < NP, we can't do a full generation.
            # For simplicity, if remaining < NP, we just do a random search or stop.
            # Given budget 500 and NP 10, we can do exactly 49 full generations after init.
            # 10 + 49*10 = 500. Perfect fit.
            
            for i in range(n_evals_this_gen):
                # Select three distinct random indices different from i
                # Simple approach: pick 3 random indices, ensure they are not i and distinct
                r1, r2, r3 = i, i, i
                while len(set([r1, r2, r3, i])) < 4:
                    r1 = rng.integers(0, NP)
                    r2 = rng.integers(0, NP)
                    r3 = rng.integers(0, NP)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                f_trial = func(trial)
                remaining -= 1
                
                # Selection
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # If we ran out of budget mid-generation, we stop.
            if remaining <= 0:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
