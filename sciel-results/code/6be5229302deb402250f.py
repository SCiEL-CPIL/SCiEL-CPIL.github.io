import random
import time
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 0..255."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(aps, colors):
    """Count monochromatic 4-APs using numpy for speed."""
    colors_arr = np.array(colors, dtype=np.uint8)
    count = 0
    for a, b, c, d in aps:
        if colors_arr[a] == colors_arr[b] == colors_arr[c] == colors_arr[d]:
            count += 1
    return count

def _count_mono_fast(aps, colors):
    """Faster count using precomputed indices."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search(aps, colors, max_iter=200, deadline=None):
    """Greedy local search: flip bits that reduce mono-AP count."""
    colors = list(colors)
    n = len(colors)
    current_score = _count_mono(aps, colors)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping bit i
            colors[i] = 1 - colors[i]
            new_score = _count_mono(aps, colors)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                colors[i] = 1 - colors[i]  # revert
    return colors, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Build all 4-APs
    aps = _build_aps()
    n_aps = len(aps)
    
    # Initial population: diverse starting points
    random.seed(42)
    pop_size = 30
    population = []
    
    # Seed with various structured patterns
    patterns = [
        [i % 2 for i in range(N)],                          # alternating
        [(i // 2) % 2 for i in range(N)],                   # blocks of 2
        [(i // 4) % 2 for i in range(N)],                   # blocks of 4
        [int(bin(i).count('1') % 2) for i in range(N)],     # Thue-Morse-like
        [((i * 7) % 3) % 2 for i in range(N)],              # modular
        [((i * 13) % 5) % 2 for i in range(N)],             # modular
    ]
    
    for p in patterns:
        population.append(list(p))
    
    # Add random individuals
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(N)])
    
    # Evaluate initial population
    scores = [_count_mono(aps, ind) for ind in population]
    
    best_idx = scores.index(min(scores))
    best_individual = list(population[best_idx])
    best_score = scores[best_idx]
    
    # Genetic algorithm with local search
    gen = 0
    while time.time() < deadline - 30.0:  # leave 30s for final local search
        gen += 1
        
        # Sort by fitness
        order = sorted(range(pop_size), key=lambda i: scores[i])
        
        # Elitism: keep top 2
        new_pop = [list(population[order[0]]), list(population[order[1]])]
        
        # Crossover and mutation to fill population
        while len(new_pop) < pop_size:
            if time.time() > deadline - 30.0:
                break
            # Tournament selection
            i1, i2 = random.sample(range(pop_size), 2)
            p1 = population[order[i1]] if scores[order[i1]] <= scores[order[i2]] else population[order[i2]]
            i3, i4 = random.sample(range(pop_size), 2)
            p2 = population[order[i3]] if scores[order[i3]] <= scores[order[i4]] else population[order[i4]]
            
            # Uniform crossover
            child = [p1[j] if random.random() < 0.5 else p2[j] for j in range(N)]
            
            # Mutation
            for j in range(N):
                if random.random() < 0.05:
                    child[j] = 1 - child[j]
            
            new_pop.append(child)
        
        # Evaluate new population
        new_scores = [_count_mono(aps, ind) for ind in new_pop]
        
        # Update best
        for i in range(pop_size):
            if new_scores[i] < best_score:
                best_score = new_scores[i]
                best_individual = list(new_pop[i])
        
        # Apply local search to top few individuals
        for i in range(min(3, pop_size)):
            idx = order[i]
            if time.time() > deadline - 30.0:
                break
            improved_ind, improved_score = _local_search(aps, new_pop[idx], max_iter=50, deadline=deadline - 30.0)
            if improved_score < new_scores[idx]:
                new_pop[idx] = improved_ind
                new_scores[idx] = improved_score
                if improved_score < best_score:
                    best_score = improved_score
                    best_individual = list(improved_ind)
        
        population = new_pop
        scores = new_scores
    
    # Final local search on best individual
    final_individual, final_score = _local_search(aps, best_individual, max_iter=500, deadline=deadline)
    if final_score < best_score:
        best_individual = final_individual
        best_score = final_score
    
    return best_individual

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
