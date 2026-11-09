import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 256

    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    aps_arr = np.array(aps, dtype=np.int32)
    
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p]
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)

    def count_mono(color):
        cols = color[aps_arr]
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    def flip_delta(color, p):
        others = pos_others[p]
        if others.shape[0] == 0:
            return 0
        c_p = color[p]
        c_new = 1 - c_p
        others_colors = color[others]
        is_mono_now = np.all(others_colors == c_p, axis=1)
        is_mono_new = np.all(others_colors == c_new, axis=1)
        return int(np.sum(is_mono_new) - np.sum(is_mono_now))
    
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = bin(i).count('1') % 2
        return color
    
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color

    def simulated_annealing(color):
        color = color.copy()
        current_score = count_mono(color)
        best_color = color.copy()
        best_score = current_score
        
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.995
        
        T = T_start
        
        while time.time() - start_time < deadline - 2 and T > T_end:
            p = np.random.randint(n)
            delta = flip_delta(color, p)
            
            if delta < 0:
                accept = True
            else:
                accept = (np.random.rand() < np.exp(-delta / T))
            
            if accept:
                color[p] = 1 - color[p]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
            
            T *= cooling_rate
            
        return best_color, best_score

    def population_annealing(startings, pop_size=16):
        # Initialize population
        population = []
        scores = []
        
        # Use provided startings if available, otherwise random
        if startings:
            for s in startings[:pop_size]:
                c = s.copy()
                population.append(c)
                scores.append(count_mono(c))
            # Fill remaining with random perturbations of best starting
            if len(population) < pop_size:
                best_start_idx = np.argmin(scores)
                for _ in range(pop_size - len(population)):
                    c = population[best_start_idx].copy()
                    flip_mask = np.random.rand(n) < 0.1
                    c[flip_mask] = 1 - c[flip_mask]
                    population.append(c)
                    scores.append(count_mono(c))
        else:
            for _ in range(pop_size):
                c = np.random.randint(0, 2, n).astype(np.int8)
                population.append(c)
                scores.append(count_mono(c))
        
        best_color = population[np.argmin(scores)].copy()
        best_score = min(scores)
        
        T_start = 5.0
        T_end = 0.05
        cooling_rate = 0.998 # Slower cooling for population to allow more exploration
        
        # Each individual has its own temperature
        temps = np.full(pop_size, T_start)
        
        while time.time() - start_time < deadline - 2 and np.any(temps > T_end):
            # Perform one step for each individual
            for i in range(pop_size):
                if temps[i] <= T_end:
                    continue
                
                c = population[i]
                p = np.random.randint(n)
                delta = flip_delta(c, p)
                
                if delta < 0:
                    accept = True
                else:
                    accept = (np.random.rand() < np.exp(-delta / temps[i]))
                
                if accept:
                    c[p] = 1 - c[p]
                    scores[i] += delta
                    if scores[i] < best_score:
                        best_score = scores[i]
                        best_color = c.copy()
                
                temps[i] *= cooling_rate
            
            # Periodic selection/mutation to maintain diversity
            # Every 50 steps, replace the worst 20% with mutated best
            # We don't have a step counter, so we use time or just do it less frequently
            # Let's say we do it when the temperature of the worst individual drops below half initial
            # Or simpler: just run the loop. The "population" aspect is the parallel trajectories.
            # To add "Vector Annealing" flavor, we can occasionally swap best/worst or mutate.
            # Let's add a simple mutation step for the worst individuals every ~100ms
            if time.time() % 0.1 < (1.0 / 100.0): # Roughly every 100ms
                # Find indices of worst 2 individuals
                worst_indices = np.argsort(scores)[-2:]
                best_idx = np.argmin(scores)
                
                for w_idx in worst_indices:
                    # Mutate the best individual
                    new_c = population[best_idx].copy()
                    # Flip 5 random bits
                    flip_pos = np.random.choice(n, size=5, replace=False)
                    new_c[flip_pos] = 1 - new_c[flip_pos]
                    population[w_idx] = new_c
                    scores[w_idx] = count_mono(new_c)
                    temps[w_idx] = T_start # Reset temperature for new individual
        
        return best_color, best_score

    startings = [thue_morse(n)]
    for bs in [2, 4, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(5):
        startings.append(random_coloring())
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=15, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)

    scored_startings = []
    for sc in startings:
        sc_score = count_mono(sc)
        scored_startings.append((sc_score, sc))
    scored_startings.sort(key=lambda x: x[0])
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # Use population annealing for the main search
    # Pass the best startings to initialize the population
    if time.time() - start_time < deadline - 5:
        # Use top 5 startings to seed the population
        seed_startings = [sc for _, sc in scored_startings[:5]]
        best_color, best_score = population_annealing(seed_startings, pop_size=16)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_color = best_color.copy()

    # If we have time, run more population annealing rounds starting from the best found
    if time.time() - start_time < deadline - 5 and best_overall_color is not None:
        while time.time() - start_time < deadline - 2:
            # Use the current best to seed the next population round
            # Add some random perturbations to the best to create diversity in the seed
            seeds = []
            base = best_overall_color
            seeds.append(base.copy())
            for _ in range(4):
                perturbed = base.copy()
                flip_mask = np.random.rand(n) < 0.05
                perturbed[flip_mask] = 1 - perturbed[flip_mask]
                seeds.append(perturbed)
            
            best_color, best_score = population_annealing(seeds, pop_size=16)
            
            if best_score < best_overall_score:
                best_overall_score = best_score
                best_overall_color = best_color.copy()
            else:
                # If no improvement in a full population run, we might be stuck.
                # Break to save time for other potential strategies? 
                # Or just continue? The population annealing itself has internal time checks.
                # Since population_annealing runs until deadline-2, it will consume the rest of the time.
                break

    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
