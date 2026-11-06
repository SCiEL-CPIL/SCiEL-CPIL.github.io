"""Improved program using structured initialization + simulated annealing."""
import numpy as np
import random

def count_mono_4ap(colors, n=256):
    """Count monochromatic 4-term arithmetic progressions in colors[0..n-1]."""
    count = 0
    # colors is a list/array of length n, indexed 0..n-1 corresponding to 1..n
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            c0 = colors[a]
            if c0 == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    
    # Try multiple structured initializations and pick the best
    # Then run simulated annealing to improve
    
    def make_block_coloring(block_size):
        """Color in blocks: first block_size are 0, next block_size are 1, etc."""
        colors = []
        for i in range(n):
            colors.append((i // block_size) % 2)
        return colors
    
    def make_mod_coloring(mod):
        """Color by i mod mod, then map to 0/1 based on parity of (i mod mod)."""
        colors = []
        for i in range(n):
            colors.append((i % mod) % 2)
        return colors
    
    # Generate candidate initial colorings
    candidates = []
    
    # Block sizes
    for bs in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128]:
        if bs <= n:
            candidates.append(make_block_coloring(bs))
    
    # Mod-based
    for mod in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64]:
        if mod <= n:
            candidates.append(make_mod_coloring(mod))
    
    # Random initializations
    for _ in range(20):
        candidates.append([random.randint(0, 1) for _ in range(n)])
    
    # Find best initial coloring
    best_colors = None
    best_count = float('inf')
    for cand in candidates:
        c = count_mono_4ap(cand, n)
        if c < best_count:
            best_count = c
            best_colors = cand[:]
    
    # Simulated annealing to improve
    colors = best_colors[:]
    current_count = best_count
    
    # Precompute all 4-APs for efficiency
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d_pos) in enumerate(ap_list):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d_pos].append(idx)
    
    # Simulated annealing
    import time
    start_time = time.time()
    time_limit = 110  # seconds, leave some margin
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99995
    
    best_colors_sa = colors[:]
    best_count_sa = current_count
    
    while time.time() - start_time < time_limit:
        # Pick a random position to flip
        pos = random.randint(0, n - 1)
        old_val = colors[pos]
        new_val = 1 - old_val
        
        # Compute delta in count
        delta = 0
        for ap_idx in pos_to_aps[pos]:
            a, b, c, d_pos = ap_list[ap_idx]
            # Check if this AP is currently monochromatic
            is_mono_before = (colors[a] == colors[b] == colors[c] == colors[d_pos])
            # After flip, check again (only pos changes)
            # We need to check all four positions
            vals = [colors[a], colors[b], colors[c], colors[d_pos]]
            # Replace the value at pos
            if a == pos:
                vals[0] = new_val
            elif b == pos:
                vals[1] = new_val
            elif c == pos:
                vals[2] = new_val
            elif d_pos == pos:
                vals[3] = new_val
            is_mono_after = (vals[0] == vals[1] == vals[2] == vals[3])
            
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
        
        # Accept or reject
        if delta <= 0:
            colors[pos] = new_val
            current_count += delta
            if current_count < best_count_sa:
                best_count_sa = current_count
                best_colors_sa = colors[:]
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < np.exp(-delta / T):
                colors[pos] = new_val
                current_count += delta
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_colors_sa
