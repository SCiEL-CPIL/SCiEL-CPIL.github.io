import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 130

    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for j in range(4):
            pos = aps[idx][j]
            pos_aps[pos].append((idx, j))
    
    def count_mono_np(color_arr):
        """Count monochromatic 4-APs using numpy."""
        c = color_arr[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def delta_flip(color, pos):
        """Return change in mono count if we flip position pos."""
        old_color = color[pos]
        new_color = 1 - old_color
        delta = 0
        for ap_idx, j in pos_aps[pos]:
            a, b, c, d = aps[ap_idx]
            # Check if all 4 are same color
            if color[a] == color[b] == color[c] == color[d]:
                # Currently monochromatic
                if old_color == color[a]:
                    # Flipping breaks this AP
                    delta -= 1
                else:
                    # This shouldn't happen since old_color is color[pos]
                    pass
            else:
                # Not currently monochromatic
                # After flip, check if it becomes monochromatic
                # The other 3 positions must all be new_color
                others_match = True
                for k in range(4):
                    if k != j:
                        if color[aps[ap_idx][k]] != new_color:
                            others_match = False
                            break
                if others_match:
                    delta += 1
        return delta
    
    def local_search(color, max_iters=500):
        """Greedy local search: flip bits that reduce mono count."""
        color = list(color)
        cur_count = count_mono_np(np.array(color, dtype=np.int8))
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            best_delta = 0
            best_pos = -1
            
            for pos in range(n):
                d = delta_flip(color, pos)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
            
            if best_pos >= 0:
                color[best_pos] = 1 - color[best_pos]
                cur_count += best_delta
                improved = True
        return color, cur_count
    
    def simulated_annealing(color, max_iters=5000, t0=5.0, t1=0.001):
        """Simulated annealing with delta-based flip evaluation."""
        color = list(color)
        cur_count = count_mono_np(np.array(color, dtype=np.int8))
        best_color = color[:]
        best_count = cur_count
        t = t0
        cooling = (t1 / t0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            pos = random.randint(0, n - 1)
            d = delta_flip(color, pos)
            new_count = cur_count + d
            if new_count < cur_count or (d > 0 and random.random() < np.exp(-d / max(t, 1e-10))):
                color[pos] = 1 - color[pos]
                cur_count = new_count
                if cur_count < best_count:
                    best_count = cur_count
                    best_color = color[:]
            t *= cooling
        
        return best_color, best_count
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def random_color(n):
        return [random.randint(0, 1) for _ in range(n)]
    
    # Initialize population with diverse starting points
    pop_size = 20
    population = []
    
    # Thue-Morse and its complement
    tm = thue_morse(n)
    population.append(tm[:])
    population.append([1 - x for x in tm])
    
    # Structured initializations with various block sizes
    for block in [1, 2, 3, 4, 5, 7, 8, 16]:
        population.append([(i // block) % 2 for i in range(n)])
    
    # Random initializations
    while len(population) < pop_size:
        population.append(random_color(n))
    
    # Evaluate initial population
    scores = []
    for ind in population:
        scores.append(count_mono_np(np.array(ind, dtype=np.int8)))
    
    best_idx = min(range(pop_size), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_color = population[best_idx][:]
    
    # Main search loop: alternating GA, local search, and SA
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        
        # Sort by fitness
        order = sorted(range(pop_size), key=lambda i: scores[i])
        
        # Elitism: keep top 3
        new_pop = [population[order[0]][:], population[order[1]][:], population[order[2]][:]]
        new_scores = [scores[order[0]], scores[order[1]], scores[order[2]]]
        
        # Generate offspring
        while len(new_pop) < pop_size and time.time() < deadline - 10:
            # Tournament selection
            def tournament():
                cands = random.sample(range(pop_size), 3)
                return min(cands, key=lambda i: scores[i])
            
            p1 = population[tournament()]
            p2 = population[tournament()]
            
            # Crossover: uniform
            child = []
            for i in range(n):
                if random.random() < 0.5:
                    child.append(p1[i])
                else:
                    child.append(p2[i])
            
            # Mutation
            mut_rate = 0.03
            for i in range(n):
                if random.random() < mut_rate:
                    child[i] = 1 - child[i]
            
            new_pop.append(child)
            new_scores.append(count_mono_np(np.array(child, dtype=np.int8)))
        
        population = new_pop
        scores = new_scores
        
        # Local search on best few individuals
        for i in range(min(5, pop_size)):
            if time.time() < deadline - 10:
                polished, sc = local_search(population[i], max_iters=100)
                population[i] = polished
                scores[i] = sc
        
        # Simulated annealing on one random individual to explore
        if time.time() < deadline - 15:
            sa_idx = random.randint(0, pop_size - 1)
            sa_color, sa_score = simulated_annealing(population[sa_idx], max_iters=3000)
            population[sa_idx] = sa_color
            scores[sa_idx] = sa_score
        
        # Update best
        cur_best = min(scores)
        if cur_best < best_score:
            best_score = cur_best
            best_idx_local = scores.index(cur_best)
            best_color = population[best_idx_local][:]
    
    # Final local search on best
    if time.time() < deadline - 5:
        polished, sc = local_search(best_color, max_iters=500)
        if sc < best_score:
            best_color = polished
            best_score = sc
    
    # Final SA polish
    if time.time() < deadline - 3:
        sa_color, sa_score = simulated_annealing(best_color, max_iters=5000, t0=10.0, t1=0.0001)
        if sa_score < best_score:
            best_color = sa_color
            best_score = sa_score
    
    # Final local search after SA
    if time.time() < deadline - 1:
        polished, sc = local_search(best_color, max_iters=300)
        if sc < best_score:
            best_color = polished
            best_score = sc
    
    return best_color
