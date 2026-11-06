"""Rewritten program with Max-SAT-inspired local search using O(1) delta updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256

def count_mono_4ap_fast(coloring):
    """Count monochromatic 4-term APs in a 0/1 coloring of 1..256."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N):
        max_a = N - 3 * d
        if max_a < 1:
            break
        seq1 = c[:max_a]
        seq2 = c[d:d + max_a]
        seq3 = c[2 * d:2 * d + max_a]
        seq4 = c[3 * d:3 * d + max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += int(np.sum(mono))
    return count

# Precompute all 4-term APs as tuples of 4 indices (0-indexed)
AP_LIST = []
for d in range(1, N):
    max_a = N - 3 * d
    for a in range(1, max_a + 1):
        # 0-indexed positions: a-1, a-1+d, a-1+2d, a-1+3d
        i1 = a - 1
        i2 = i1 + d
        i3 = i2 + d
        i4 = i3 + d
        AP_LIST.append((i1, i2, i3, i4))

NUM_APS = len(AP_LIST)

# For each position, precompute the list of APs it participates in, with the other 3 indices
AP_POS = [[] for _ in range(N)]
for ap_idx, (i1, i2, i3, i4) in enumerate(AP_LIST):
    for pos in (i1, i2, i3, i4):
        others = tuple(x for x in (i1, i2, i3, i4) if x != pos)
        AP_POS[pos].append((ap_idx, others))

def compute_ap_mono(coloring):
    """Return a boolean array of length NUM_APS where True = monochromatic."""
    mono = np.zeros(NUM_APS, dtype=bool)
    for idx, (i1, i2, i3, i4) in enumerate(AP_LIST):
        c0 = coloring[i1]
        if c0 == coloring[i2] == coloring[i3] == coloring[i4]:
            mono[idx] = True
    return mono

def delta_flip(pos, color, coloring, ap_mono):
    """Compute delta in count of mono APs if we flip position pos to color."""
    current_color = coloring[pos]
    if current_color == color:
        return 0
    # For each AP containing pos, determine if flipping changes mono status
    delta = 0
    for ap_idx, others in AP_POS[pos]:
        c_o1 = coloring[others[0]]
        c_o2 = coloring[others[1]]
        c_o3 = coloring[others[2]]
        # Currently: is this AP mono?
        currently_mono = ap_mono[ap_idx]
        # After flip: all four must be color
        after_mono = (c_o1 == color) and (c_o2 == color) and (c_o3 == color)
        if currently_mono and not after_mono:
            delta -= 1
        elif not currently_mono and after_mono:
            delta += 1
    return delta

def local_search_1bit(coloring, max_iters=200, deadline=None):
    """Hill-climbing with O(1) delta evaluation for single-bit flips."""
    coloring = np.array(coloring, dtype=np.int8)
    ap_mono = compute_ap_mono(coloring)
    score = int(np.sum(ap_mono))
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        best_pos = -1
        best_color = -1
        best_delta = 0
        for pos in range(N):
            # Try flipping to 0
            d0 = delta_flip(pos, 0, coloring, ap_mono)
            if d0 < best_delta:
                best_delta = d0
                best_pos = pos
                best_color = 0
            # Try flipping to 1
            d1 = delta_flip(pos, 1, coloring, ap_mono)
            if d1 < best_delta:
                best_delta = d1
                best_pos = pos
                best_color = 1
        if best_pos == -1:
            break
        # Apply the best flip
        old_color = coloring[best_pos]
        coloring[best_pos] = best_color
        # Update ap_mono for affected APs
        for ap_idx, others in AP_POS[best_pos]:
            c0 = coloring[best_pos]
            ap_mono[ap_idx] = (c0 == coloring[others[0]] == coloring[others[1]] == coloring[others[2]])
        score += best_delta
    
    return coloring, score

def generate_initial_colorings():
    """Generate diverse initial colorings."""
    colorings = []
    
    # Random uniform
    for _ in range(30):
        colorings.append(np.random.randint(0, 2, N))
    
    # Block-based patterns
    for block_size in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64]:
        if block_size <= N:
            colorings.append(np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8))
    
    # Alternating with noise
    for noise_level in [0.05, 0.1, 0.15, 0.2, 0.3]:
        base = np.array([i % 2 for i in range(N)], dtype=np.int8)
        noise = np.random.random(N) < noise_level
        colorings.append(np.where(noise, 1 - base, base).astype(np.int8))
    
    # Van der Waerden-inspired mod patterns
    for mod in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        colorings.append(np.array([1 if (i % mod) < mod // 2 else 0 for i in range(N)], dtype=np.int8))
    
    # Greedy construction
    greedy = greedy_construct()
    colorings.append(greedy)
    
    return colorings

def greedy_construct():
    """Greedy: build coloring one position at a time."""
    coloring = np.full(N, -1, dtype=np.int8)
    for j in range(N):
        best_color = 0
        best_count = float('inf')
        for color in (0, 1):
            new_count = 0
            for d in range(1, (j // 3) + 1):
                a_idx = j - 3 * d
                if a_idx < 0:
                    break
                pos1 = a_idx
                pos2 = a_idx + d
                pos3 = a_idx + 2 * d
                if coloring[pos1] != -1 and coloring[pos2] != -1 and coloring[pos3] != -1:
                    if coloring[pos1] == color and coloring[pos2] == color and coloring[pos3] == color:
                        new_count += 1
            if new_count < best_count:
                best_count = new_count
                best_color = color
        coloring[j] = best_color
    return coloring

def iterated_local_search(best_coloring, best_score, deadline):
    """ILS: perturb best solution, then local search."""
    current_best = best_coloring.copy()
    current_score = best_score
    
    while time.time() < deadline - 5:
        # Perturb: flip 3-8 random bits
        c_perturbed = current_best.copy()
        num_flips = random.randint(3, 8)
        for _ in range(num_flips):
            pos = random.randint(0, N - 1)
            c_perturbed[pos] = 1 - c_perturbed[pos]
        
        # Local search
        c_ls, score_ls = local_search_1bit(c_perturbed, max_iters=150, deadline=deadline - 5)
        
        if score_ls < current_score:
            current_score = score_ls
            current_best = c_ls.copy()
    
    return current_best, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Phase 1: Multi-start local search
    best_coloring = None
    best_score = float('inf')
    
    initial_colorings = generate_initial_colorings()
    
    for i, c_init in enumerate(initial_colorings):
        if time.time() > deadline - 20:
            break
        
        c_ls, score_ls = local_search_1bit(c_init, max_iters=150, deadline=deadline - 10)
        
        if score_ls < best_score:
            best_score = score_ls
            best_coloring = c_ls.copy()
    
    # Phase 2: Iterated Local Search
    if time.time() < deadline - 10:
        best_coloring, best_score = iterated_local_search(best_coloring, best_score, deadline)
    
    # Phase 3: Final polish with more iterations
    if time.time() < deadline - 5:
        c_final, score_final = local_search_1bit(best_coloring, max_iters=300, deadline=deadline - 2)
        if score_final < best_score:
            best_score = score_final
            best_coloring = c_final
    
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
