import random
import time
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_to_aps(aps):
    """For each position, store (ap_index, position_in_ap) for all APs it belongs to."""
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append((idx, 0))
        pos_to_aps[b].append((idx, 1))
        pos_to_aps[c].append((idx, 2))
        pos_to_aps[d].append((idx, 3))
    return pos_to_aps

def _score_full_np(aps_arr, colors_np):
    """Count monochromatic 4-APs using numpy vectorization."""
    c0 = colors_np[aps_arr[:, 0]]
    c1 = colors_np[aps_arr[:, 1]]
    c2 = colors_np[aps_arr[:, 2]]
    c3 = colors_np[aps_arr[:, 3]]
    return int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))

def _local_search_incremental(aps, pos_to_aps, colors, max_passes=200, deadline=None):
    """
    Greedy local search using incremental delta scoring.
    Optimized with precomputed neighbor info.
    """
    colors = list(colors)
    n = len(colors)
    
    # Precompute AP positions as numpy array for fast access
    ap_positions = np.array(aps, dtype=np.int32)
    
    # Current mono status of each AP
    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())
    
    # Precompute for each position: the list of (ap_idx, other_positions_tuple)
    # For faster delta computation
    pos_info = []
    for i in range(n):
        info = []
        for ap_idx, pos_in_ap in pos_to_aps[i]:
            a, b, c, d = aps[ap_idx]
            if pos_in_ap == 0:
                others = (b, c, d)
            elif pos_in_ap == 1:
                others = (a, c, d)
            elif pos_in_ap == 2:
                others = (a, b, d)
            else:
                others = (a, b, c)
            info.append((ap_idx, others))
        pos_info.append(info)
    
    for _pass in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            old = colors[i]
            new = 1 - old
            colors[i] = new
            
            # Compute delta
            delta = 0
            for ap_idx, others in pos_info[i]:
                was_mono = mono_flags[ap_idx]
                # Check if others are all same color as each other
                o1, o2, o3 = others
                others_same = (colors[o1] == colors[o2] == colors[o3])
                if others_same:
                    # If old color matched others, flipping breaks mono
                    # If old color didn't match, flipping creates mono
                    if was_mono:
                        delta -= 1
                    else:
                        delta += 1
                    mono_flags[ap_idx] = 1 if delta != 0 or not was_mono else 0
                    # Actually need to be more careful
                # Recompute properly
                a, b, c, d = aps[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono
            
            if delta < 0:
                current_score += delta
                improved = True
            else:
                colors[i] = old
                for ap_idx, others in pos_info[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
        
        if not improved:
            break
    
    return colors, current_score

def _simulated_annealing(aps, pos_to_aps, colors, start_temp=5.0, end_temp=0.05, max_iter=8000, deadline=None):
    """
    Simulated annealing with 1-bit and 2-bit flip moves.
    """
    colors = list(colors)
    n = len(colors)
    
    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())
    best_score = current_score
    best_colors = colors[:]
    
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        temp *= cooling
        
        # Choose move type: 75% single flip, 25% 2-bit flip
        if random.random() < 0.75:
            i = random.randint(0, n - 1)
            old = colors[i]
            colors[i] = 1 - old
            
            delta = 0
            for ap_idx, pos_in_ap in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono
            
            new_score = current_score + delta
            if new_score < current_score or random.random() < np.exp(-max(delta, 0) / max(temp, 1e-10)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                colors[i] = old
                for ap_idx, pos_in_ap in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
        else:
            # 2-bit flip
            i1 = random.randint(0, n - 1)
            i2 = random.randint(0, n - 1)
            while i2 == i1:
                i2 = random.randint(0, n - 1)
            
            old1 = colors[i1]
            old2 = colors[i2]
            colors[i1] = 1 - old1
            colors[i2] = 1 - old2
            
            # Compute delta for both flips together
            affected = set()
            for ap_idx, _ in pos_to_aps[i1]:
                affected.add(ap_idx)
            for ap_idx, _ in pos_to_aps[i2]:
                affected.add(ap_idx)
            
            delta = 0
            for ap_idx in affected:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono
            
            new_score = current_score + delta
            if new_score < current_score or random.random() < np.exp(-max(delta, 0) / max(temp, 1e-10)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                colors[i1] = old1
                colors[i2] = old2
                for ap_idx in affected:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    
    return best_colors, best_score

def _ga_population_init(pop_size):
    """Initialize diverse GA population."""
    population = []
    
    # 1. Random
    for _ in range(pop_size // 4):
        population.append([random.randint(0, 1) for _ in range(N)])
    
    # 2. Block patterns
    for block_size in [2, 3, 4, 5, 8, 16, 32]:
        population.append([(i // block_size) % 2 for i in range(N)])
    
    # 3. Thue-Morse
    population.append([bin(i).count('1') % 2 for i in range(N)])
    
    # 4. Striped patterns
    for period in [3, 5, 7, 11, 13, 17]:
        pattern = []
        for i in range(N):
            pos_in_period = i % period
            pattern.append(0 if pos_in_period < period // 2 else 1)
        population.append(pattern)
    
    # 5. Quadratic residue-like patterns
    for p in [3, 5, 7, 11, 13]:
        pop = []
        for i in range(N):
            is_res = any((x * x) % p == i % p for x in range(p))
            pop.append(1 if is_res else 0)
        population.append(pop)
    
    # 6. 2D grid-inspired
    grid_pop = []
    for i in range(N):
        row = i // 16
        col = i % 16
        grid_pop.append((row + col) % 2)
    population.append(grid_pop)
    
    # 7. Another 2D pattern
    grid_pop2 = []
    for i in range(N):
        row = i // 16
        col = i % 16
        grid_pop2.append(((row // 2) + (col // 2)) % 2)
    population.append(grid_pop2)
    
    # Pad with random
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(N)])
    
    return population[:pop_size]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    aps = _build_aps()
    pos_to_aps = _build_pos_to_aps(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    best_score = float('inf')
    best_pop = None
    
    pop_size = 40
    population = _ga_population_init(pop_size)
    
    # Evaluate initial population using numpy
    scores = []
    for pop in population:
        pop_np = np.array(pop, dtype=np.int32)
        scores.append(_score_full_np(aps_arr, pop_np))
    
    order = sorted(range(len(population)), key=lambda i: scores[i])
    population = [population[i] for i in order]
    scores = [scores[i] for i in order]
    
    best_score = scores[0]
    best_pop = population[0][:]
    
    # Local search on top candidates
    for i in range(min(8, len(population))):
        if time.time() > deadline - 50:
            break
        improved_pop, s = _local_search_incremental(aps, pos_to_aps, population[i], max_passes=150, deadline=deadline - 50)
        if s < best_score:
            best_score = s
            best_pop = improved_pop[:]
    
    # Multi-start SA from best few solutions
    for i in range(min(4, len(population))):
        if time.time() > deadline - 30:
            break
        sa_pop, s = _simulated_annealing(aps, pos_to_aps, population[i], start_temp=5.0, end_temp=0.05, max_iter=5000, deadline=deadline - 30)
        if s < best_score:
            best_score = s
            best_pop = sa_pop[:]
        
        # Local search after SA
        if time.time() < deadline - 25:
            ls_pop, s = _local_search_incremental(aps, pos_to_aps, best_pop, max_passes=200, deadline=deadline - 25)
            if s < best_score:
                best_score = s
                best_pop = ls_pop[:]
    
    # Main GA loop
    mutation_rate = 0.06
    elitism = 3
    gen = 0
    
    while time.time() < deadline - 30:
        gen += 1
        
        def tournament_select():
            idx1 = random.randint(0, len(population) - 1)
            idx2 = random.randint(0, len(population) - 1)
            return population[idx1][:] if scores[idx1] <= scores[idx2] else population[idx2][:]
        
        new_population = []
        
        # Elitism
        for i in range(elitism):
            new_population.append(population[i][:])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 25:
                break
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Uniform crossover
            child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(N)]
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            new_population.append(child)
        
        # Evaluate new population using numpy
        new_scores = []
        for pop in new_population:
            pop_np = np.array(pop, dtype=np.int32)
            new_scores.append(_score_full_np(aps_arr, pop_np))
        
        order = sorted(range(len(new_population)), key=lambda i: new_scores[i])
        population = [new_population[i] for i in order]
        scores = [new_scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_pop = population[0][:]
        
        # Local search on top few each generation
        for i in range(min(4, len(population))):
            if time.time() > deadline - 20:
                break
            improved_pop, s = _local_search_incremental(aps, pos_to_aps, population[i], max_passes=80, deadline=deadline - 20)
            if s < best_score:
                best_score = s
                best_pop = improved_pop[:]
        
        # Periodically inject diversity
        if gen % 8 == 0 and time.time() < deadline - 35:
            for _ in range(3):
                new_ind = [random.randint(0, 1) for _ in range(N)]
                new_ind_np = np.array(new_ind, dtype=np.int32)
                s = _score_full_np(aps_arr, new_ind_np)
                if s < scores[-1]:
                    population[-1] = new_ind
                    scores[-1] = s
    
    # Final intensive local search
    if time.time() < deadline - 10:
        final_pop, s = _local_search_incremental(aps, pos_to_aps, best_pop, max_passes=400, deadline=deadline - 8)
        if s < best_score:
            best_score = s
            best_pop = final_pop[:]
    
    # Final SA polish
    if time.time() < deadline - 5:
        sa_pop, s = _simulated_annealing(aps, pos_to_aps, best_pop, start_temp=2.0, end_temp=0.01, max_iter=3000, deadline=deadline - 3)
        if s < best_score:
            best_score = s
            best_pop = sa_pop[:]
        
        # One more local search
        if time.time() < deadline - 1:
            final_pop, s = _local_search_incremental(aps, pos_to_aps, best_pop, max_passes=100, deadline=deadline - 0.5)
            if s < best_score:
                best_score = s
                best_pop = final_pop[:]
    
    return best_pop
