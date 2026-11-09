"""Rewritten optimizer using Differential Evolution with adaptive restarts."""
import numpy as np


class Optimizer:
    """Differential Evolution based continuous optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Clip helper
        def clip(x):
            return np.clip(x, -5.0, 5.0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        n = self.dim
        
        # DE Parameters
        pop_size = 15
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # 1. Initialize Population
        pop = np.empty((pop_size, n))
        fvals = np.empty(pop_size)
        
        for i in range(pop_size):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, n)
            pop[i] = x
            fvals[i] = func(x)
            calls_made += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = x.copy()
        
        if calls_made >= self.budget:
            return best_x, best_f
            
        # 2. DE Main Loop
        no_improve_gens = 0
        max_no_improve = 15
        
        while calls_made < self.budget:
            # Process one generation
            gen_improved = False
            
            for i in range(pop_size):
                if calls_made >= self.budget:
                    break
                
                # Pick 3 distinct indices for mutation
                # Ensure we don't pick i
                candidates = np.delete(np.arange(pop_size), i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # Mutation: v = pop[r1] + F * (pop[r2] - pop[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = clip(v)
                
                # Crossover: Binomial
                j_rand = rng.integers(0, n)
                u = np.empty(n)
                # Vectorized crossover for speed
                mask = rng.random(n) < CR
                mask[j_rand] = True # Ensure at least one gene from v
                u = np.where(mask, v, pop[i])
                
                # Evaluate
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu <= fvals[i]:
                    pop[i] = u
                    fvals[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        gen_improved = True
            
            if calls_made >= self.budget:
                break
            
            if gen_improved:
                no_improve_gens = 0
            else:
                no_improve_gens += 1
                
            if no_improve_gens > max_no_improve:
                # Restart: Replace population with new points around best_x
                # Use a radius that is initially large and shrinks
                # Simple approach: constant radius 1.0
                radius = 1.0
                for i in range(pop_size):
                    if calls_made >= self.budget:
                        break
                    x = best_x + rng.normal(0, radius, n)
                    x = clip(x)
                    pop[i] = x
                    fvals[i] = func(x)
                    calls_made += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = x.copy()
                no_improve_gens = 0

        return best_x, best_f
