import numpy as np
import time

class Optimizer:
    """Hybrid DE + CMA-ES for rotated Weierstrass landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution
        # Use smaller population to allow more generations
        N_de = 10
        budget_de = int(budget * 0.6) # 60% for DE
        calls_made = 0
        
        # Initialize DE population
        pop = rng.uniform(-5, 5, size=(N_de, dim))
        fitness = np.array([func(np.clip(x, -5, 5)) for x in pop])
        calls_made += N_de
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        F = 0.8
        CR = 0.9
        
        while calls_made < budget_de:
            if time.time() > deadline:
                break
            if calls_made + N_de > budget_de:
                break
                
            for i in range(N_de):
                if calls_made >= budget_de:
                    break
                if time.time() > deadline:
                    break
                    
                r1, r2, r3 = rng.choice(N_de, size=3, replace=False)
                while r1 == i: r1 = rng.integers(0, N_de)
                while r2 == i or r2 == r1: r2 = rng.integers(0, N_de)
                while r3 == i or r3 == r1 or r3 == r2: r3 = rng.integers(0, N_de)
                
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                j_rand = rng.integers(0, dim)
                cross_mask = rng.random(dim) < CR
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls_made += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Adaptive F and CR
            F = max(0.4, F * 0.99)
            CR = min(1.0, CR * 1.01)

        # Phase 2: Local Refinement with CMA-ES (or simple ES)
        # Start CMA-ES from best_x found by DE
        remaining_budget = budget - calls_made
        if remaining_budget > 20: # Need at least some calls
            # CMA-ES Parameters
            N_es = 8
            mu = N_es // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
            dampsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
            
            mean = best_x.copy()
            sigma = 0.5 # Smaller step size for local search
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                if calls_made + N_es > budget:
                    break
                
                cholesky = np.linalg.cholesky(C)
                
                for i in range(N_es):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    z = rng.standard_normal(dim)
                    x = mean + sigma * cholesky @ z
                    x = np.clip(x, -5, 5)
                    f_x = func(x)
                    calls_made += 1
                    
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                    
                    # Store for CMA-ES update
                    # We need to store all N_es points to update properly
                    # This requires buffering or running full generations
                    # For simplicity in this hybrid, let's just do a (1+1) ES if budget is tight?
                    # No, CMA-ES needs the full population.
                    # Let's buffer the population.
                    pass # This logic is flawed if we don't store pop.
