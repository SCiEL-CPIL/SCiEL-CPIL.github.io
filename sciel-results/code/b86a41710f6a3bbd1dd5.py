import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # Store as 0-indexed tuples
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)

    # For each index, precompute the list of AP indices it belongs to
    # and for each such AP, which other 3 indices are in it
    index_to_aps = [[] for _ in range(N)]
    for ap_idx, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            others = [x for x in (i1, i2, i3, i4) if x != idx]
            index_to_aps[idx].append((ap_idx, others))

    def count_mono(color):
        """Count monochromatic 4-APs using NumPy."""
        cols = color[aps_array]
        is_mono = np.all(cols == cols[:, 0:1], axis=1)
        return np.sum(is_mono).item()

    def get_delta(color, idx, new_val):
        """Compute change in mono count if we flip color[idx] to new_val."""
        old_val = color[idx]
        if old_val == new_val:
            return 0
        delta = 0
        for ap_idx, others in index_to_aps[idx]:
            # Check if the other 3 are all the same color
            o0, o1, o2 = others
            if color[o0] == color[o1] == color[o2]:
                # The other 3 are monochromatic with color c_others
                c_others = color[o0]
                if old_val == c_others:
                    # Currently mono. If new_val != c_others, it becomes non-mono.
                    if new_val != c_others:
                        delta -= 1
                else:
                    # Currently non-mono (idx is different from others).
                    # If new_val == c_others, it becomes mono.
                    if new_val == c_others:
                        delta += 1
        return delta

    def local_search(color, max_iter=1000):
        """Hill climbing with random order to avoid cycles."""
        current_score = count_mono(color)
        improved = True
        iter_count = 0
        rng_ls = random.Random()
        while improved and iter_count < max_iter:
            improved = False
            indices = list(range(N))
            rng_ls.shuffle(indices)
            for idx in indices:
                new_val = 1 - color[idx]
                delta = get_delta(color, idx, new_val)
                if delta < 0:
                    color[idx] = new_val
                    current_score += delta
                    improved = True
            iter_count += 1
        return current_score

    def sa_optimize(color, T_start=5.0, T_min=0.05, alpha=0.9995, max_steps=30000):
        """Simulated Annealing with single-bit flips and periodic block-reversal kicks."""
        current_score = count_mono(color)
        best_score = current_score
        best_color = color.copy()
        
        T = T_start
        rng = random.Random()
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Periodic kick: reverse a random block to escape local minima
            if step % 100 == 0 and T > T_min * 10:
                block_len = rng.randint(5, 30)
                if block_len < N:
                    start_idx = rng.randint(0, N - block_len)
                    end_idx = start_idx + block_len - 1
                    
                    # Perform the reversal
                    segment = color[start_idx:end_idx+1].copy()
                    color[start_idx:end_idx+1] = segment[::-1]
                    
                    # Recalculate count (fast now)
                    new_score = count_mono(color)
                    delta = new_score - current_score
                    
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_color = color.copy()
                    else:
                        # Revert
                        color[start_idx:end_idx+1] = segment
            
            # Standard SA step: single bit flip
            idx = rng.randint(0, N - 1)
            new_val = 1 - color[idx]
            delta = get_delta(color, idx, new_val)
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                color[idx] = new_val
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
            
            T *= alpha
            if T < T_min:
                T = T_min
            
            # Early exit if we hit a very low count
            if best_score <= 5:
                break
        
        return best_score, best_color

    # Generate good initial configurations
    def random_init():
        return np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)

    def block_init(block_size):
        return np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)

    def stripe_init(period):
        return np.array([(i % period) < period // 2 for i in range(N)]).astype(np.int8)

    candidates = []
    
    # Block patterns
    for bs in [2, 3, 4, 5, 6, 7, 8, 16]:
        candidates.append(block_init(bs))
    
    # Stripe patterns
    for p in [3, 5, 7, 9, 11, 13]:
        candidates.append(stripe_init(p))
    
    # Random inits
    for _ in range(20):
        candidates.append(random_init())
    
    # Evaluate and refine candidates with local search
    scored = []
    for c in candidates:
        c = c.copy()
        sc = local_search(c, max_iter=100)
        scored.append((sc, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Track best overall
    best_score = scored[0][0]
    best_color = scored[0][1].copy()
    
    # Genetic Algorithm + SA Hybrid
    rng = random.Random()
    
    # Initialize population from top candidates
    pop_size = 20
    population = []
    for i in range(min(pop_size, len(scored))):
        population.append(scored[i][1].copy())
    # Fill remaining with random
    while len(population) < pop_size:
        population.append(random_init())
    
    # GA parameters
    mutation_rate = 0.05
    elite_count = 3
    
    while time.time() < deadline - 10.0:
        # Evaluate population
        scored_pop = []
        for ind in population:
            sc = count_mono(ind)
            scored_pop.append((sc, ind))
        
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_color = scored_pop[0][1].copy()
        
        # Keep elites
        new_population = [ind.copy() for _, ind in scored_pop[:elite_count]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5.0:
                break
            
            # Tournament selection
            def tournament_select(pop, scores, k=3):
                candidates_t = rng.sample(range(len(pop)), min(k, len(pop)))
                best_idx = min(candidates_t, key=lambda i: scores[i])
                return pop[best_idx].copy()
            
            scores_list = [s for s, _ in scored_pop]
            parent1 = tournament_select(population, scores_list)
            parent2 = tournament_select(population, scores_list)
            
            # Uniform crossover
            child = np.array([
                parent1[i] if rng.random() < 0.5 else parent2[i] 
                for i in range(N)
            ], dtype=np.int8)
            
            # Mutation
            for i in range(N):
                if rng.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Quick local search on child
            local_search(child, max_iter=20)
            
            new_population.append(child)
        
        population = new_population
        
        # Apply SA to the best individual in population
        if time.time() < deadline - 10.0:
            best_in_pop = population[0].copy()
            sa_score, sa_color = sa_optimize(best_in_pop, T_start=3.0, T_min=0.02, alpha=0.9995, max_steps=10000)
            
            if sa_score < best_score:
                best_score = sa_score
                best_color = sa_color.copy()
            
            # Replace worst in population with SA result
            population[-1] = sa_color.copy()
    
    # Final exhaustive local search
    if time.time() < deadline - 1.0:
        local_search(best_color, max_iter=1000)
        best_score = count_mono(best_color)
    
    return [int(x) for x in best_color]
