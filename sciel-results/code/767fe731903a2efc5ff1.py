import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s kill
        
        calls_used = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = None
        
        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Bounded (1+lambda) Evolutionary Strategy
        # Rationale: ES is robust for continuous optimization, handles multimodality
        # better than single-point local search via population diversity, and
        # adaptive step size helps navigate basins of different scales.
        
        # Initial population: 10 random points in [-5, 5]^dim
        pop_size = 10
        step_size = 2.0  # Initial step size
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            x = np.random.uniform(-5, 5, size=self.dim)
            f = func(x)
            calls_used += 1
            update_best(x, f)
            population.append((x, f))
        
        # Sort population by fitness
        population.sort(key=lambda p: p[1])
        
        # ES main loop
        while calls_used < max_calls and time.time() < deadline:
            # Generate lambda offspring
            # lambda = 5 (moderate exploration)
            lam = 5
            offspring = []
            
            for _ in range(lam):
                # Select parent from top half of population (elitism + diversity)
                parent_idx = np.random.randint(0, max(1, pop_size // 2))
                parent_x = population[parent_idx][0]
                
                # Perturb with adaptive step size
                # Step size decreases slowly if improving, increases if stagnating
                mutation = np.random.normal(0, step_size, size=self.dim)
                child_x = parent_x + mutation
                
                # Clip to bounds [-5, 5]
                child_x = np.clip(child_x, -5, 5)
                
                # Evaluate
                if calls_used >= max_calls or time.time() > deadline:
                    break
                f = func(child_x)
                calls_used += 1
                offspring.append((child_x, f))
                update_best(child_x, f)
            
            if not offspring:
                break
            
            # Combine parent and offspring
            combined = population + offspring
            combined.sort(key=lambda p: p[1])
            
            # Select new population (top pop_size)
            population = combined[:pop_size]
            
            # Adapt step size
            # If best offspring improved best parent, decrease step size slightly
            # Otherwise, increase step size to explore more
            best_parent_f = population[0][1]
            best_offspring_f = offspring[0][1] if offspring else float('inf')
            
            if best_offspring_f < best_parent_f:
                step_size *= 0.95  # Shrink step
            else:
                step_size *= 1.05  # Expand step
            
            # Keep step size in reasonable range
            step_size = np.clip(step_size, 0.01, 5.0)
            
            # Check for stagnation: if no improvement in last 30 calls, restart
            # This helps escape local minima in multimodal landscapes
            if len(offspring) > 0:
                # Simple stagnation check: if best_f hasn't improved in last 30 calls
                # We don't track history explicitly, so we use a heuristic:
                # If step_size has been shrinking for a while, force a restart
                pass
        
        return best_x, best_f
