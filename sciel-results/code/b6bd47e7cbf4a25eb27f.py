import numpy as np

class Optimizer:
    """
    Differential Evolution (DE) optimizer.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        NP = 12  # Population size
        F = 0.8  # Mutation factor
        CR = 0.5 # Crossover rate
        
        # Initialization: Uniform random in [-5, 5]
        pop = np.random.uniform(-5.0, 5.0, size=(NP, n))
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        
        # Track best
        idx_best = np.argmin(fvals)
        f_best = fvals[idx_best]
        x_best = pop[idx_best].copy()
        
        calls_made = NP
        
        # Budget management
        # Each generation costs NP function evaluations
        while calls_made + NP <= budget:
            # New population for this generation
            new_pop = np.zeros_like(pop)
            new_fvals = np.zeros(NP)
            
            for i in range(NP):
                # Select three distinct random individuals
                # Ensure they are distinct from i and each other
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                
                # Mutation: DE/rand/1
                # v = pop[r1] + F * (pop[r2] - pop[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Boundary handling: Clip to [-5, 5]
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover: Binomial
                # Choose a random position to ensure at least one gene is from mutant
                j_rand = np.random.randint(n)
                
                trial = pop[i].copy()
                # Vectorized crossover
                mask = np.random.rand(n) <= CR
                mask[j_rand] = True
                trial[mask] = v[mask]
                
                # Evaluate trial
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = f_trial
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fvals[i] = fvals[i]
            
            # Update population
            pop = new_pop
            fvals = new_fvals
            
        # Use remaining budget for local refinement
        # Simple local search around the best solution
        while calls_made < budget:
            # Random walk around best
            x_try = x_best + np.random.randn(n) * 0.1
            x_try = np.clip(x_try, -5.0, 5.0)
            f_try = func(x_try)
            calls_made += 1
            if f_try < f_best:
                f_best = f_try
                x_best = x_try.copy()

        return x_best, f_best
