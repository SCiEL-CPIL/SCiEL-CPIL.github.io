import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in dev, but task says hidden seeded shift. 
        # Wait, the task says "hidden seeded shift". The optimizer's internal RNG should probably be fixed or seeded by instance?
        # The task says "For each instance... evaluator creates Optimizer... and calls it once."
        # If I use a fixed seed in Optimizer, it's deterministic. 
        # Usually, in these black-box tasks, you can use any seed.
        
        # Reserve budget for local search
        local_search_budget = min(100, budget // 5)
        de_budget = budget - local_search_budget
        
        pop_size = 20
        # Ensure pop_size is even? Not strictly necessary for DE/rand/1.
        
        # Initialize
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size
        
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        F = 0.8
        CR = 0.9
        
        while calls_made < de_budget:
            # Generate trials
            for i in range(pop_size):
                if calls_made >= de_budget:
                    break
                
                # Select r1, r2, r3 distinct from i and each other
                while True:
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    if len(set([r1, r2, r3, i])) == 4:
                        break
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        # Local Search Phase
        # Use the best_x found by DE as the starting point for local search.
        # We have local_search_budget calls left.
        
        # Define a wrapper to count calls if necessary, but scipy has maxfev.
        # We need to be careful: scipy might make more calls than expected if not constrained.
        # maxfev constrains the number of function evaluations.
        
        # We have 'budget - calls_made' calls left.
        remaining_calls = budget - calls_made
        
        if remaining_calls > 0:
            # Use Nelder-Mead
            # options: maxfev
            # We must ensure we don't exceed the total budget.
            # scipy.optimize.minimize might call func more times than maxfev? No, maxfev is strict.
            # However, if the optimizer terminates early, we might have unused budget.
            # We can just call it.
            
            # It's safer to implement a simple local search or use scipy.
            # Let's use scipy.
            
            # Note: func must be a callable that takes x.
            # scipy expects func(x) where x is 1-D array.
            
            # We should probably try multiple local searches if we have budget?
            # Or just one from the best?
            # Given Büche-Rastrigin, the best DE point is likely in the correct basin.
            
            res = minimize(
                func, 
                best_x, 
                method='Nelder-Mead',
                options={'maxfev': remaining_calls, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            # res.fun is the best value found by local search
            # res.x is the best point
            # We must check if res.fun is better than best_f
            # Note: minimize might fail or return nan? Unlikely.
            
            if res.success or res.fun < best_f:
                best_f = res.fun
                best_x = res.x
            else:
                # If minimize failed or didn't improve, keep DE best
                pass
                
        return best_x, best_f
