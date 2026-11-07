import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size: 5 to 10 times dim is typical. 
        # For dim=5, let's use 20.
        NP = max(10, 5 * dim)
        
        # Initialize population uniformly in [-5, 5]
        pop = np.random.uniform(-5, 5, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        calls = NP
        
        # DE Parameters
        F = 0.8  # Mutation scale
        CR = 0.5  # Crossover rate
        
        while calls < budget:
            # If we are running low on budget, stop
            if calls + NP > budget:
                # Use remaining budget for local refinement or just stop?
                # Better to stop if we can't complete a generation, 
                # or do a partial generation?
                # Let's just break if we can't do a full generation to keep it simple.
                # Or we can do a local search on best_x with remaining budget.
                break
            
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros_like(fitness)
            
            for i in range(NP):
                # Select 3 distinct random indices
                while True:
                    r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                    if r1 != r2 and r2 != r3 and r1 != r3 and i not in [r1, r2, r3]:
                        break
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            pop = new_pop
            fitness = new_fitness
            
            # Optional: Adapt F if no improvement?
            # For simplicity, keep F constant.
            
        # If we have remaining budget, do a local Nelder-Mead or simple gradient-free refinement?
        # Given the "Early progress" scoring, refining the best point is crucial.
        # Let's use the remaining budget for a simple local search (hill climbing) around best_x.
        
        if calls < budget:
            # Simple local search: random perturbations decreasing in size
            current_x = best_x.copy()
            current_f = best_f
            step_size = 0.5
            remaining = budget - calls
            
            while remaining > 0:
                # Try a few random steps
                for _ in range(min(remaining, 10)):
                    if remaining <= 0: break
                    dx = np.random.randn(dim) * step_size
                    new_x = np.clip(current_x + dx, -5, 5)
                    new_f = func(new_x)
                    remaining -= 1
                    if new_f < current_f:
                        current_x = new_x
                        current_f = new_f
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                # Reduce step size
                step_size *= 0.7
                if step_size < 1e-4:
                    break

        return best_x, best_f
