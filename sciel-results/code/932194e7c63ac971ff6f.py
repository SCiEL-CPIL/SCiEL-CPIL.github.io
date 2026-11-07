import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315  # 15s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        dim = self.dim
        
        # Multi-Island DE Parameters
        num_islands = 5
        pop_per_island = 10
        total_pop = num_islands * pop_per_island
        
        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9 # Crossover rate
        
        # Migration
        migration_interval = 10
        gen_count = 0
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            
            # ERROR_REPAIR OPERATOR: 
            # The task states func clips, but we can also repair here to ensure 
            # our internal state tracks the effective point if we were to use 
            # gradients or local steps. For DE, we just evaluate.
            # However, the 'error_repair' concept in the directive asks if it 
            # changes the chance of improvement. In DE, we generate points 
            # that might be out of bounds. Clipping them (repair) is standard.
            # We let func handle the clipping as per contract, but we track 
            # the clipped version for our population state to avoid 
            # "wasted" diversity in the out-of-bounds region.
            
            val = func(x)
            
            # For bookkeeping, we store the clipped point as the "true" position
            # in our population to keep the search focused inside the box.
            x_clipped = np.clip(x, -5.0, 5.0)
            
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Initialize Islands
        # Each island gets a random population
        # We use a fixed seed for reproducibility in logic, but random for actual run
        # Actually, let's just use np.random.
        
        islands = []
        island_fitness = []
        
        for i in range(num_islands):
            # Random initialization in [-5, 5]
            pop = np.random.uniform(-5, 5, size=(pop_per_island, dim))
            fits = np.full(pop_per_island, np.inf)
            
            for j in range(pop_per_island):
                try:
                    fits[j] = wrapper(pop[j])
                except (ValueError, TimeoutError):
                    break
            
            # If budget ran out during init, return best
            if calls >= max_calls or time.time() > deadline:
                if best_x is not None:
                    return best_x, best_f
                return np.zeros(dim), np.inf

            islands.append(pop)
            island_fitness.append(fits)

        # Main Loop
        while True:
            if calls >= max_calls or time.time() > deadline:
                break
            
            gen_count += 1
            
            # Migration
            if gen_count % migration_interval == 0:
                # Simple migration: swap the worst individual of each island 
                # with the best individual of the next island (circular)
                for i in range(num_islands):
                    # Find worst in island i
                    worst_idx_i = np.argmax(island_fitness[i])
                    # Find best in island (i+1) % num_islands
                    next_island = (i + 1) % num_islands
                    best_idx_next = np.argmin(island_fitness[next_island])
                    
                    # Swap
                    # The migrant takes the spot of the worst
                    islands[i][worst_idx_i] = islands[next_island][best_idx_next].copy()
                    island_fitness[i][worst_idx_i] = island_fitness[next_island][best_idx_next]
                    
                    # Note: The donor island loses its best? No, usually migration 
                    # copies. But to maintain population size and diversity, 
                    # let's just copy the best from neighbor to worst slot.
                    # Actually, standard DE migration often replaces the worst 
                    # with a random or best from neighbor. Let's stick to 
                    # replacing worst with best from neighbor (copy).
                    # The source island keeps its best.
            
            # Evolution on each island
            for i in range(num_islands):
                if calls >= max_calls or time.time() > deadline:
                    break
                
                pop = islands[i]
                fits = island_fitness[i]
                
                # Sort population by fitness
                idx = np.argsort(fits)
                pop = pop[idx]
                fits = fits[idx]
                
                # Generate offspring
                new_pop = np.copy(pop)
                new_fits = np.copy(fits)
                
                for j in range(pop_per_island):
                    if calls >= max_calls or time.time() > deadline:
                        break
                    
                    # Select 3 distinct random individuals from the island
                    # Ensure they are distinct from j
                    r1, r2, r3 = j, j, j
                    while r1 == j or r1 == r2 or r1 == r3:
                        r1 = np.random.randint(0, pop_per_island)
                        r2 = np.random.randint(0, pop_per_island)
                        r3 = np.random.randint(0, pop_per_island)
                    
                    # DE/rand/1
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover
                    u = np.copy(pop[j])
                    # Random crossover
                    j_rand = np.random.randint(0, dim)
                    for k in range(dim):
                        if np.random.rand() < CR or k == j_rand:
                            u[k] = v[k]
                        else:
                            u[k] = pop[j][k]
                    
                    # Error Repair: Clip to box
                    u = np.clip(u, -5.0, 5.0)
                    
                    try:
                        f_u = wrapper(u)
                    except (ValueError, TimeoutError):
                        break
                    
                    # Selection
                    if f_u < fits[j]:
                        new_pop[j] = u
                        new_fits[j] = f_u
                
                # Update island state
                islands[i] = new_pop
                island_fitness[i] = new_fits

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf

        return best_x, best_f
