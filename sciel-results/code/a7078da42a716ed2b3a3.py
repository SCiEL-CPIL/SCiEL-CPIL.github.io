"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # Population size for DE
        # Standard is 5-10 * dim. For dim=5, 25-50.
        # Let's use 20 to balance diversity and budget usage.
        pop_size = 20 
        
        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9 # Crossover rate

        rng = np.random.default_rng(42)

        # Initialize population
        # Uniform in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, n))
        
        # Evaluate initial population
        # We have `budget` calls. Initial pop uses `pop_size` calls.
        # Remaining budget for generations.
        # Each generation uses `pop_size` calls (one per individual).
        
        if budget <= pop_size:
             # Edge case: budget too small for DE
             # Just do random search
             best_x = np.zeros(n)
             best_f = float('inf')
             calls_made = 0
             for _ in range(budget):
                 if time.time() > deadline:
                     break
                 x = rng.uniform(-5, 5, n)
                 f = func(x)
                 calls_made += 1
                 if f < best_f:
                     best_f = f
                     best_x = x.copy()
             return best_x

        f_pop = np.empty(pop_size)
        calls_made = 0
        
        for i in range(pop_size):
            f_pop[i] = func(pop[i])
            calls_made += 1
            
        # Track best
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]

        # Main DE loop
        # Each generation costs `pop_size` function evaluations.
        # Total generations we can afford: (budget - calls_made) // pop_size
        
        gen = 0
        while calls_made + pop_size <= budget:
            if time.time() > deadline:
                break
                
            gen += 1
            new_pop = np.empty((pop_size, n))
            new_f_pop = np.empty(pop_size)
            
            for i in range(pop_size):
                # Select 3 distinct random indices
                # a, b, c != i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover: trial vector u
                u = np.empty(n)
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                        
                # Evaluate trial
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= f_pop[i]:
                    new_pop[i] = u
                    new_f_pop[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                else:
                    new_pop[i] = pop[i]
                    new_f_pop[i] = f_pop[i]
                    
            pop = new_pop
            f_pop = new_f_pop
            
            # Check for stagnation or convergence
            # If best hasn't improved in a while, we might want to inject diversity
            # But standard DE with fixed params usually works well enough for 500 calls.
            # We could add a restart mechanism if variance is low.
            # For now, keep it simple to ensure we use the budget efficiently.
            
            # Optional: Local refinement around the best point if we have time/budget left?
            # DE is global, but Rosenbrock has a narrow valley.
            # However, DE with CR=0.9 and F=0.8 is quite aggressive.
            # Let's stick to pure DE for the main loop.
            
        # If we have leftover budget (budget % pop_size != 0), use it for local search around best
        remaining = budget - calls_made
        if remaining > 0:
             # Simple local random search around best_x
             # Shrink radius
             radius = 1.0
             for _ in range(remaining):
                 if time.time() > deadline:
                     break
                 # Perturb best_x
                 x_trial = best_x + rng.normal(0, radius, n)
                 x_trial = np.clip(x_trial, -5, 5)
                 f_trial = func(x_trial)
                 calls_made += 1
                 if f_trial < best_f:
                     best_f = f_trial
                     best_x = x_trial.copy()
                     # Shrink radius slightly on improvement
                     radius *= 0.9
                 else:
                     # Shrink radius slightly on failure
                     radius *= 0.99
                     if radius < 1e-5:
                         radius = 1e-5

        return best_x

# EVOLVE-BLOCK-END
