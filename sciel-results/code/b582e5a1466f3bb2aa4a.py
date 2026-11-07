"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Parallel Multi-Scale Differential Evolution Optimizer.
    
    Maintains three populations:
    1. Global (N=10): Explores the whole domain.
    2. Basin (N=5): Focuses on promising regions.
    3. Fine (N=2): Polishes the current best.
    
    Uses adaptive sigma and migration between islands.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(42)
        
        # Population sizes
        N_GLOBAL = 10
        N_BASIN = 5
        N_FINE = 2
        
        # Initialize Populations
        # Global: Uniform in [-5, 5]
        pop_global = rng.uniform(-5, 5, (N_GLOBAL, dim))
        # Basin: Uniform in [-2, 2]
        pop_basin = rng.uniform(-2, 2, (N_BASIN, dim))
        # Fine: Uniform in [-0.5, 0.5]
        pop_fine = rng.uniform(-0.5, 0.5, (N_FINE, dim))

        # Storage for function values
        f_global = np.full(N_GLOBAL, np.inf)
        f_basin = np.full(N_BASIN, np.inf)
        f_fine = np.full(N_FINE, np.inf)

        x_best = None
        f_best = float('inf')
        calls = 0
        
        # Helper to evaluate and update best
        def safe_eval(x):
            nonlocal calls, x_best, f_best
            if calls >= budget or time.time() > deadline:
                return None
            try:
                fx = func(x)
            except:
                return None
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # Initial Evaluation
        for i in range(N_GLOBAL):
            fx = safe_eval(pop_global[i])
            if fx is not None:
                f_global[i] = fx
        for i in range(N_BASIN):
            fx = safe_eval(pop_basin[i])
            if fx is not None:
                f_basin[i] = fx
        for i in range(N_FINE):
            fx = safe_eval(pop_fine[i])
            if fx is not None:
                f_fine[i] = fx

        # Adaptive Parameters
        sigma_global = 1.5
        sigma_basin = 0.5
        sigma_fine = 0.1
        F = 0.8  # Mutation factor
        CR = 0.5 # Crossover rate

        # Migration counters
        gen = 0
        migration_interval = 5  # Migrate every 5 generations

        while calls < budget:
            if time.time() > deadline:
                break
            
            # --- Global Population Update ---
            for i in range(N_GLOBAL):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Select two distinct parents
                a, b = rng.choice(N_GLOBAL, 2, replace=False)
                while a == i or b == i:
                    a, b = rng.choice(N_GLOBAL, 2, replace=False)
                
                # DE/2/rand mutation
                v = pop_global[i] + F * (pop_global[a] - pop_global[b])
                # Add noise for exploration
                v = v + sigma_global * rng.standard_normal(dim)
                v = np.clip(v, -5, 5)
                
                fx = safe_eval(v)
                if fx is None:
                    continue
                
                # Selection
                if fx < f_global[i]:
                    pop_global[i] = v
                    f_global[i] = fx

            # --- Basin Population Update ---
            for i in range(N_BASIN):
                if calls >= budget or time.time() > deadline:
                    break
                
                a, b = rng.choice(N_BASIN, 2, replace=False)
                while a == i or b == i:
                    a, b = rng.choice(N_BASIN, 2, replace=False)
                
                v = pop_basin[i] + F * (pop_basin[a] - pop_basin[b])
                v = v + sigma_basin * rng.standard_normal(dim)
                v = np.clip(v, -5, 5)
                
                fx = safe_eval(v)
                if fx is None:
                    continue
                
                if fx < f_basin[i]:
                    pop_basin[i] = v
                    f_basin[i] = fx

            # --- Fine Population Update ---
            for i in range(N_FINE):
                if calls >= budget or time.time() > deadline:
                    break
                
                # For small pop, just perturb best
                # Use global best as reference if available
                if x_best is not None:
                    v = x_best + sigma_fine * rng.standard_normal(dim)
                else:
                    v = pop_fine[i] + sigma_fine * rng.standard_normal(dim)
                
                v = np.clip(v, -5, 5)
                
                fx = safe_eval(v)
                if fx is None:
                    continue
                
                if fx < f_fine[i]:
                    pop_fine[i] = v
                    f_fine[i] = fx

            # --- Migration and Adaptation ---
            gen += 1
            if gen % migration_interval == 0:
                # Move best from Global to Basin
                idx_g_best = np.argmin(f_global)
                if f_global[idx_g_best] < f_basin[np.argmax(f_basin)]:
                    # Replace worst in basin with best from global
                    idx_b_worst = np.argmax(f_basin)
                    pop_basin[idx_b_worst] = pop_global[idx_g_best].copy()
                    f_basin[idx_b_worst] = f_global[idx_g_best]
                
                # Move best from Basin to Fine
                idx_b_best = np.argmin(f_basin)
                if f_basin[idx_b_best] < f_fine[np.argmax(f_fine)]:
                    idx_f_worst = np.argmax(f_fine)
                    pop_fine[idx_f_worst] = pop_basin[idx_b_best].copy()
                    f_fine[idx_f_worst] = f_basin[idx_b_best]

                # Adapt sigmas
                # If global is improving, keep sigma; else reduce slightly
                # Simple heuristic: reduce sigma as we converge
                sigma_global *= 0.98
                sigma_basin *= 0.98
                sigma_fine *= 0.98
                
                # Prevent underflow
                sigma_global = max(sigma_global, 0.1)
                sigma_basin = max(sigma_basin, 0.01)
                sigma_fine = max(sigma_fine, 0.001)

        return x_best, f_best

# EVOLVE-BLOCK-END
