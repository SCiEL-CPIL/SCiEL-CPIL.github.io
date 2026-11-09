"""Optimizer for BBOB f23 (Katsuura) using Derandomized Evolutionary Strategy (DES)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Derandomized Evolutionary Strategy (DES) for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        # We use 30 points for initial spread
        n_lhs = 30
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            if time.time() > deadline:
                return x_global_best, fx_global_best

        # Phase 2: Derandomized Evolutionary Strategy (DES)
        # Use a fixed population of deterministic mutation vectors
        # This provides systematic exploration without random restarts
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # Define population size for DES
        # A larger population allows for more systematic exploration
        pop_size = 20
        
        # Create deterministic mutation vectors using a structured grid
        # We use a set of orthogonal directions scaled appropriately
        # For 5D, we can use the standard basis vectors and their combinations
        
        # Generate deterministic mutation directions
        # Use a combination of standard basis and Hadamard-like patterns
        mutation_vectors = []
        
        # Standard basis vectors
        for i in range(dim):
            v = np.zeros(dim)
            v[i] = 1.0
            mutation_vectors.append(v)
            v[i] = -1.0
            mutation_vectors.append(v)
            
        # Add diagonal directions
        for i in range(dim):
            for j in range(i+1, dim):
                v = np.zeros(dim)
                v[i] = 1.0
                v[j] = 1.0
                mutation_vectors.append(v / np.sqrt(2))
                v[i] = -1.0
                v[j] = -1.0
                mutation_vectors.append(v / np.sqrt(2))
                
        # Truncate to population size
        mutation_vectors = mutation_vectors[:pop_size]
        
        # Initial step size
        sigma = 1.5
        min_sigma = 0.01
        max_sigma = 5.0
        
        # Population: list of (x, fx) tuples
        # Initialize population with global best and its mutations
        population = [(x_global_best.copy(), fx_global_best)]
        
        # Generate initial population members
        for i in range(1, pop_size):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                return x_global_best, fx_global_best
                
            # Mutate global best using deterministic direction
            mut_dir = mutation_vectors[(i - 1) % len(mutation_vectors)]
            x_new = x_global_best + sigma * mut_dir
            x_new = np.clip(x_new, bounds, upper)
            
            fx_new = func(x_new)
            calls_made += 1
            update_best(x_new, fx_new)
            population.append((x_new.copy(), fx_new))
            
        # Main DES loop
        while calls_made < budget and time.time() < deadline:
            # Sort population by fitness
            population.sort(key=lambda item: item[1])
            
            # Keep the best half of the population
            elite_size = max(1, pop_size // 2)
            population = population[:elite_size]
            
            # Generate offspring from the best individuals
            new_population = []
            
            # Add the best individual as a seed
            best_x, best_fx = population[0]
            new_population.append((best_x.copy(), best_fx))
            
            # Generate mutations from the best individuals
            for i in range(len(population)):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                    
                x_parent, fx_parent = population[i]
                
                # Select mutation direction based on index
                # Use a rotating set of directions for diversity
                dir_idx = (calls_made + i) % len(mutation_vectors)
                mut_dir = mutation_vectors[dir_idx]
                
                # Adaptive step size: smaller for better individuals
                # Linear scaling from max_sigma to min_sigma based on rank
                rank_factor = i / max(1, len(population) - 1)
                current_sigma = max_sigma - rank_factor * (max_sigma - min_sigma)
                
                x_child = x_parent + current_sigma * mut_dir
                x_child = np.clip(x_child, bounds, upper)
                
                fx_child = func(x_child)
                calls_made += 1
                update_best(x_child, fx_child)
                new_population.append((x_child.copy(), fx_child))
                
                # If we have enough offspring, stop
                if len(new_population) >= pop_size:
                    break
                    
            # Replace population with new offspring
            population = new_population[:pop_size]
            
            # Check for convergence
            if len(population) > 1:
                if abs(population[0][1] - population[-1][1]) < 1e-12:
                    # Convergence detected, reduce step size
                    sigma = max(sigma * 0.5, min_sigma)
                    if sigma <= min_sigma:
                        break

        # Phase 3: Fine-grained local search on global best
        remaining = budget - calls_made
        if remaining > 5 and x_global_best is not None:
            # Use a simple coordinate descent for final polishing
            step_size = 0.1
            x_current = x_global_best.copy()
            fx_current = fx_global_best
            
            for _ in range(min(remaining - 1, 20)):
                if calls_made >= budget - 1:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                    
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 1:
                        break
                    if time.time() > deadline:
                        return x_global_best, fx_global_best
                        
                    # Try positive step
                    x_trial = x_current.copy()
                    x_trial[i] += step_size
                    x_trial = np.clip(x_trial, bounds, upper)
                    fx_trial = func(x_trial)
                    calls_made += 1
                    
                    if fx_trial < fx_current:
                        x_current = x_trial
                        fx_current = fx_trial
                        update_best(x_current, fx_current)
                        improved = True
                        continue
                    
                    # Try negative step
                    x_trial = x_current.copy()
                    x_trial[i] -= step_size
                    x_trial = np.clip(x_trial, bounds, upper)
                    fx_trial = func(x_trial)
                    calls_made += 1
                    
                    if fx_trial < fx_current:
                        x_current = x_trial
                        fx_current = fx_trial
                        update_best(x_current, fx_current)
                        improved = True
                        continue
                        
                if not improved:
                    # No improvement, reduce step size
                    step_size *= 0.5
                    if step_size < 0.001:
                        break

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
