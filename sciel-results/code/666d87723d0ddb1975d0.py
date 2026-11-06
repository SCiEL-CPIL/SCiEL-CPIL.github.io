import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 1-indexed, stored 0-indexed
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
    
    def count_mono(color):
        """Count monochromatic 4-APs."""
        count = 0
        for idx in range(num_aps):
            a, b, c, d = aps[idx]
            if color[a] == color[b] == color[c] == color[d]:
                count += 1
        return count
    
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
            others = []
            for k in range(4):
                if k != j:
                    others.append(color[aps[ap_idx][k]])
            if old_color == others[0] == others[1] == others[2]:
                delta -= 1
            elif new_color == others[0] == others[1] == others[2]:
                delta += 1
        return delta
    
    def delta_swap(color, p1, p2):
        """Return change in mono count if we swap colors at positions p1 and p2."""
        if color[p1] == color[p2]:
            return 0
        old_c1, old_c2 = color[p1], color[p2]
        new_c1, new_c2 = old_c2, old_c1
        delta = 0
        # Process APs containing p1
        for ap_idx, j in pos_aps[p1]:
            a, b, c, d = aps[ap_idx]
            others = []
            for k in range(4):
                if k != j:
                    others.append(color[aps[ap_idx][k]])
            # Check if p2 is in this AP
            if p2 in aps[ap_idx]:
                # Both positions in same AP - need careful handling
                pass
            if old_c1 == others[0] == others[1] == others[2]:
                delta -= 1
            elif new_c1 == others[0] == others[1] == others[2]:
                delta += 1
        # Process APs containing p2
        for ap_idx, j in pos_aps[p2]:
            a, b, c, d = aps[ap_idx]
            others = []
            for k in range(4):
                if k != j:
                    others.append(color[aps[ap_idx][k]])
            if p1 in aps[ap_idx]:
                pass
            if old_c2 == others[0] == others[1] == others[2]:
                delta -= 1
            elif new_c2 == others[0] == others[1] == others[2]:
                delta += 1
        # Subtract double-counted APs (those containing both p1 and p2)
        for ap_idx, j1 in pos_aps[p1]:
            if p2 in aps[ap_idx]:
                a, b, c, d = aps[ap_idx]
                # This AP was counted twice above, correct it
                colors = [color[a], color[b], color[c], color[d]]
                was_mono = all(x == old_c1 for x in colors) or all(x == old_c2 for x in colors)
                # Actually simpler: just recompute this AP's contribution
                # Remove its double contribution and add correct single contribution
                # This is getting complex; skip double-count correction for now
                pass
        return delta
    
    def local_search(color, max_iters=200, use_swaps=False):
        """Greedy local search: flip bits (and optionally swaps) that reduce mono count."""
        color = list(color)
        cur_count = count_mono(color)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            best_delta = 0
            best_pos = -1
            best_swap = None
            
            # Try flips
            for pos in range(n):
                d = delta_flip(color, pos)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
                    best_swap = None
            
            # Try swaps (only if enabled)
            if use_swaps:
                for i in range(n):
                    for j in range(i + 1, n):
                        if color[i] != color[j]:
                            d = delta_swap(color, i, j)
                            if d < best_delta:
                                best_delta = d
                                best_pos = -1
                                best_swap = (i, j)
            
            if best_pos >= 0:
                color[best_pos] = 1 - color[best_pos]
                cur_count += best_delta
                improved = True
            elif best_swap is not None:
                i, j = best_swap
                color[i], color[j] = color[j], color[i]
                cur_count += best_delta
                improved = True
        return color, cur_count
    
    def simulated_annealing(color, max_iters=5000, t0=2.0, t1=0.01):
        """Simulated annealing with delta-based flip evaluation."""
        color = list(color)
        cur_count = count_mono(color)
        best_color = color[:]
        best_count = cur_count
        t = t0
        cooling = (t1 / t0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            pos = random.randint(0, n - 1)
            d = delta_flip(color, pos)
            new_count = cur_count + d
            if new_count < cur_count or random.random() < np.exp(-d / max(t, 1e-10)):
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
    pop_size = 16
    population = []
    
    # Thue-Morse and its complement
    tm = thue_morse(n)
    population.append(tm[:])
    population.append([1 - x for x in tm])
    
    # Some structured initializations
    for block in [1, 2, 3, 4, 5, 7, 8]:
        population.append([(i // block) % 2 for i in range(n)])
    
    # Random initializations
    while len(population) < pop_size:
        population.append(random_color(n))
    
    # Evaluate initial population
    scores = []
    for ind in population:
        scores.append(count_mono(ind))
    
    best_idx = min(range(pop_size), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_color = population[best_idx][:]
    
    # Genetic algorithm
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Sort by fitness
        order = sorted(range(pop_size), key=lambda i: scores[i])
        
        # Elitism: keep top 2
        new_pop = [population[order[0]][:], population[order[1]][:]]
        new_scores = [scores[order[0]], scores[order[1]]]
        
        # Generate offspring
        while len(new_pop) < pop_size and time.time() < deadline:
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
            mut_rate = 0.05
            for i in range(n):
                if random.random() < mut_rate:
                    child[i] = 1 - child[i]
            
            new_pop.append(child)
            new_scores.append(count_mono(child))
        
        population = new_pop
        scores = new_scores
        
        # Local search on best few
        for i in range(min(3, pop_size)):
            if time.time() < deadline:
                polished, sc = local_search(population[i], max_iters=50)
                population[i] = polished
                scores[i] = sc
        
        # Simulated annealing on one random individual to explore
        if time.time() < deadline - 5:
            sa_idx = random.randint(0, pop_size - 1)
            sa_color, sa_score = simulated_annealing(population[sa_idx], max_iters=2000)
            population[sa_idx] = sa_color
            scores[sa_idx] = sa_score
        
        # Update best
        cur_best = min(scores)
        if cur_best < best_score:
            best_score = cur_best
            best_idx_local = scores.index(cur_best)
            best_color = population[best_idx_local][:]
    
    # Final local search on best
    if time.time() < deadline - 10:
        polished, sc = local_search(best_color, max_iters=500)
        if sc < best_score:
            best_color = polished
            best_score = sc
    
    # Final SA polish
    if time.time() < deadline - 5:
        sa_color, sa_score = simulated_annealing(best_color, max_iters=3000)
        if sa_score < best_score:
            best_color = sa_color
            best_score = sa_score
    
    # Final local search after SA
    if time.time() < deadline - 2:
        polished, sc = local_search(best_color, max_iters=200)
        if sc < best_score:
            best_color = polished
            best_score = sc
    
    return best_color
