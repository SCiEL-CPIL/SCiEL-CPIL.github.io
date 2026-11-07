# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget
        
        # Differential Evolution parameters
        # For dim=5, use a moderate population size
        pop_size = max(10, 4 * dim)  # 20 for dim=5
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if self._call_count >= budget or time.time() >= deadline:
                break
            fvals[i] = count_eval(population[i])
        
        if len(fvals) < pop_size:
            fvals = fvals[:len(fvals)]
            population = population[:len(fvals)]
            pop_size = len(fvals)
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]
        
        # DE with adaptive parameters and restarts
        no_improve_count = 0
        max_no_improve = 2 * pop_size
        current_F = F
        current_CR = CR
        stagnation_limit = 3 * pop_size
        
        while self._call_count + pop_size <= budget and time.time() < deadline:
            # Generate new population
            new_population = np.empty((pop_size, dim))
            new_fvals = np.empty(pop_size)
            
            for i in range(pop_size):
                if self._call_count >= budget or time.time() >= deadline:
                    break
                    
                # Select three distinct random indices different from i
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3:
                        break
                
                # Create trial vector using DE/rand/1
                v = population[r1] + current_F * (population[r2] - population[r3])
                
                # Binomial crossover
                j_rand = rng.integers(0, dim)
                u = rng.random(dim)
                trial = np.where(u < current_CR, v, population[i])
                trial[j_rand] = v[j_rand]  # Ensure at least one element from v
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = count_eval(trial)
                
                # Selection: keep the better of trial and current
                if f_trial <= fvals[i]:
                    new_population[i] = trial
                    new_fvals[i] = f_trial
                else:
                    new_population[i] = population[i]
                    new_fvals[i] = fvals[i]
            
            # Update population
            population = new_population[:len(new_fvals)]
            fvals = new_fvals
            pop_size = len(fvals)
            
            # Track best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f - 1e-12:
                best_f = fvals[idx_min]
                best_x = population[idx_min].copy()
                no_improve_count = 0
            else:
                no_improve_count += pop_size
            
            # Adaptive parameter adjustment
            if no_improve_count > stagnation_limit:
                # Restart: re-seed around best solution
                spread = max(0.5, best_f * 0.01 + 0.1)
                for i in range(pop_size):
                    if self._call_count >= budget or time.time() >= deadline:
                        break
                    perturbation = rng.uniform(-spread, spread, dim)
                    new_point = np.clip(best_x + perturbation, -5, 5)
                    f_new = count_eval(new_point)
                    population[i] = new_point
                    fvals[i] = f_new
                no_improve_count = 0
                # Reset adaptive parameters
                current_F = rng.uniform(0.5, 1.0)
                current_CR = rng.uniform(0.7, 0.95)
            
            # Occasionally adjust F and CR
            elif no_improve_count > pop_size:
                current_F = np.clip(current_F + rng.normal(0, 0.1), 0.4, 1.2)
                current_CR = np.clip(current_CR + rng.normal(0, 0.05), 0.5, 0.95)
        
        # Final local refinement using coordinate descent on best solution
        if self._call_count < budget - dim and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if self._call_count >= budget or time.time() >= deadline:
                    break
                improved = True
                while improved:
                    improved = False
                    for d in range(dim):
                        if self._call_count >= budget or time.time() >= deadline:
                            break
                        # Try positive step
                        x_try = x_local.copy()
                        x_try[d] = min(5.0, x_try[d] + step)
                        f_try = count_eval(x_try)
                        if f_try < f_local - 1e-12:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                        # Try negative step
                        if self._call_count >= budget or time.time() >= deadline:
                            break
                        x_try = x_local.copy()
                        x_try[d] = max(-5.0, x_try[d] - step)
                        f_try = count_eval(x_try)
                        if f_try < f_local - 1e-12:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                
                # Update best if improved
                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
