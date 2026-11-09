import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs as numpy arrays for vectorization
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Convert to numpy arrays for faster access
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorization."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        bad = ((c0 == c1) & (c2 == c3)) | \
              ((c0 == c2) & (c1 == c3)) | \
              ((c0 == c3) & (c1 == c2))
        return int(np.sum(bad))

    def compute_delta(colors, i, old_color, new_color):
        """Compute delta in bad count when changing colors[i] from old_color to new_color."""
        ap_indices = idx_to_aps_arr[i]
        if len(ap_indices) == 0:
            return 0
        
        sub_aps = aps[ap_indices]
        
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        pos_mask_0 = (sub_aps[:, 0] == i)
        pos_mask_1 = (sub_aps[:, 1] == i)
        pos_mask_2 = (sub_aps[:, 2] == i)
        pos_mask_3 = (sub_aps[:, 3] == i)
        
        old_bad = ((c0 == c1) & (c2 == c3)) | \
                  ((c0 == c2) & (c1 == c3)) | \
                  ((c0 == c3) & (c1 == c2))
        
        nc0 = np.where(pos_mask_0, new_color, c0)
        nc1 = np.where(pos_mask_1, new_color, c1)
        nc2 = np.where(pos_mask_2, new_color, c2)
        nc3 = np.where(pos_mask_3, new_color, c3)
        
        new_bad = ((nc0 == nc1) & (nc2 == nc3)) | \
                  ((nc0 == nc2) & (nc1 == nc3)) | \
                  ((nc0 == nc3) & (nc1 == nc2))
        
        return int(np.sum(new_bad) - np.sum(old_bad))

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Population-based search with restarts
        pop_size = 50
        max_restarts = 100
        restart_count = 0
        
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Initialize population
            population = []
            for _ in range(pop_size):
                colors = np.random.randint(0, k, n).astype(np.int32)
                cost = count_bad_colors(colors)
                population.append((cost, colors.copy()))
            
            # Sort by cost
            population.sort(key=lambda x: x[0])
            
            best_in_pop_cost, best_in_pop_colors = population[0]
            
            if best_in_pop_cost < best_k_cost:
                best_k_cost = best_in_pop_cost
                best_k_solution = best_in_pop_colors.copy()
                
            if best_k_cost == 0:
                break
                
            # Evolutionary loop
            max_gens = 100
            for gen in range(max_gens):
                if time.time() > deadline - 5:
                    break
                    
                if best_k_cost == 0:
                    break
                    
                new_population = []
                
                # Elitism: keep top 10%
                elite_count = max(1, pop_size // 10)
                for i in range(elite_count):
                    new_population.append(population[i])
                
                # Fill rest with offspring
                while len(new_population) < pop_size:
                    # Tournament selection
                    t_size = 3
                    candidates = random.sample(population, t_size)
                    parent1 = min(candidates, key=lambda x: x[0])[1]
                    parent2 = min(candidates, key=lambda x: x[0])[1]
                    
                    # Crossover: uniform with segment preservation
                    # Pick a random split point and swap segments
                    split = np.random.randint(1, n)
                    child = parent1.copy()
                    child[split:] = parent2[split:]
                    
                    # Mutation: flip some bits
                    mutation_rate = 0.05
                    for i in range(n):
                        if random.random() < mutation_rate:
                            child[i] = np.random.randint(0, k)
                    
                    cost = count_bad_colors(child)
                    new_population.append((cost, child))
                
                # Sort new population
                new_population.sort(key=lambda x: x[0])
                population = new_population
                
                best_in_pop_cost, best_in_pop_colors = population[0]
                
                if best_in_pop_cost < best_k_cost:
                    best_k_cost = best_in_pop_cost
                    best_k_solution = best_in_pop_colors.copy()
                    
                    if best_k_cost == 0:
                        break

        if best_k_cost == 0:
            return [int(c) for c in best_k_solution]
            
    # Fallback: Greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
