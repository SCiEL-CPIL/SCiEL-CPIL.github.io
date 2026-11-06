"""Program using optimized multi-start simulated annealing with diverse structured initializations."""
import numpy as np
import random
import time

def count_mono_4ap(colors, n=256):
    """Count monochromatic 4-term arithmetic progressions in colors[0..n-1]."""
    count = 0
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            c0 = colors[a]
            if c0 == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    
    # Precompute neighbors for fast delta calculation
    # neighbors[p] is a list of tuples (q1, q2, q3) for each AP containing p
    neighbors = [[] for _ in range(n)]
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            d_pos = a + 3 * d
            neighbors[a].append((b, c, d_pos))
            neighbors[b].append((a, c, d_pos))
            neighbors[c].append((a, b, d_pos))
            neighbors[d_pos].append((a, b, c))

    def make_thue_morse():
        """Thue-Morse sequence: t(n) = popcount(n) % 2."""
        colors = []
        for i in range(n):
            colors.append(bin(i).count('1') % 2)
        return colors

    def make_block_coloring(block_size):
        colors = []
        for i in range(n):
            colors.append((i // block_size) % 2)
        return colors
    
    def make_mod_coloring(mod):
        colors = []
        for i in range(n):
            colors.append((i % mod) % 2)
        return colors

    def make_xor_pattern(x):
        """Color based on XOR of block index and sub-block index."""
        colors = []
        for i in range(n):
            block = i // x
            sub = (i % x) // (x // 2)
            colors.append(block % 2 ^ sub)
        return colors
        
    def add_noise(colors, p):
        """Flip bits with probability p."""
        noisy = colors[:]
        for i in range(n):
            if random.random() < p:
                noisy[i] = 1 - noisy[i]
        return noisy

    def greedy_polish(colors, max_passes=10):
        """Flip bits that strictly reduce the count."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for pos in range(n):
                old_val = colors[pos]
                delta = 0
                for q1, q2, q3 in neighbors[pos]:
                    matches = (colors[q1] == old_val) + (colors[q2] == old_val) + (colors[q3] == old_val)
                    if matches == 3:
                        delta -= 1
                    elif matches == 0:
                        delta += 1
                if delta < 0:
                    colors[pos] = 1 - old_val
                    improved = True
        return colors

    # Generate diverse structured candidates
    candidates = []
    
    # Thue-Morse and inverted
    tm = make_thue_morse()
    candidates.append(tm)
    candidates.append([1 - c for c in tm])
    
    # Noisy Thue-Morse
    candidates.append(add_noise(tm, 0.05))
    candidates.append(add_noise(tm, 0.1))
    
    # Block sizes
    for bs in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]:
        candidates.append(make_block_coloring(bs))
        if bs >= 4:
            candidates.append(add_noise(make_block_coloring(bs), 0.05))
            
    # Mod-based
    for mod in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]:
        candidates.append(make_mod_coloring(mod))
        
    # XOR patterns
    for x in [4, 8, 16, 32, 64]:
        candidates.append(make_xor_pattern(x))
        
    # Random initializations
    for _ in range(20):
        candidates.append([random.randint(0, 1) for _ in range(n)])
        
    # Find best initial coloring
    best_colors_init = None
    best_count_init = float('inf')
    for cand in candidates:
        c = count_mono_4ap(cand, n)
        if c < best_count_init:
            best_count_init = c
            best_colors_init = cand[:]
            
    # Multi-Start Simulated Annealing
    start_time = time.time()
    time_limit = 115  # seconds
    
    global_best_colors = best_colors_init[:]
    global_best_count = best_count_init
    
    # Polish the initial best
    polished = greedy_polish(global_best_colors[:], max_passes=20)
    polished_count = count_mono_4ap(polished, n)
    if polished_count < global_best_count:
        global_best_count = polished_count
        global_best_colors = polished[:]

    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialize current state
        if start_idx == 0:
            colors = global_best_colors[:]
        else:
            # Perturb the global best by flipping a few random bits
            colors = global_best_colors[:]
            num_flips = random.randint(5, 25)
            for _ in range(num_flips):
                pos = random.randint(0, n - 1)
                colors[pos] = 1 - colors[pos]
        
        current_count = count_mono_4ap(colors, n)
        
        # SA Parameters
        T = 5.0
        T_min = 0.01
        max_iters = 200000
        cooling = (T_min / T) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            # Check time every 1000 iterations
            if i % 1000 == 0:
                if time.time() - start_time > time_limit - 2:
                    break
                
            pos = random.randint(0, n - 1)
            old_val = colors[pos]
            
            delta = 0
            for q1, q2, q3 in neighbors[pos]:
                matches = (colors[q1] == old_val) + (colors[q2] == old_val) + (colors[q3] == old_val)
                if matches == 3:
                    delta -= 1
                elif matches == 0:
                    delta += 1
            
            if delta <= 0:
                colors[pos] = 1 - old_val
                current_count += delta
                if current_count < global_best_count:
                    global_best_count = current_count
                    global_best_colors = colors[:]
            else:
                if random.random() < np.exp(-delta / T):
                    colors[pos] = 1 - old_val
                    current_count += delta
            
            T *= cooling
            if T < T_min:
                T = T_min

        # Polish the result of this SA run
        polished = greedy_polish(colors[:], max_passes=10)
        polished_count = count_mono_4ap(polished, n)
        if polished_count < global_best_count:
            global_best_count = polished_count
            global_best_colors = polished[:]

    return global_best_colors
