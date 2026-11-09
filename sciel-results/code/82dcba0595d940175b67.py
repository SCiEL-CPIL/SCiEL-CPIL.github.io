"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
TIME_LIMIT = 145  # safety margin before 150s kill

def _count_mono_4aps_vec(colors):
    """Count monochromatic 4-term APs using vectorized numpy.
    colors is a numpy array of 0/1 of length N."""
    n = len(colors)
    count = 0
    for d in range(1, (n - 3) // 3 + 1):
        # a ranges from 0 to n - 1 - 3*d
        a_max = n - 1 - 3 * d
        if a_max < 0:
            break
        # Extract the four terms for all valid a
        c0 = colors[0:a_max + 1]
        c1 = colors[d:d + a_max + 1]
        c2 = colors[2 * d:2 * d + a_max + 1]
        c3 = colors[3 * d:3 * d + a_max + 1]
        # Monochromatic means all four equal
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        count += np.sum(mono)
    return count

def _build_ap_structure():
    """Precompute all 4-term APs and build reverse index.
    Returns:
        aps: list of tuples (a, a+d, a+2d, a+3d) using 0-indexed positions
        pos_to_ap_indices: list of lists, for each position, indices of APs containing it
    """
    n = N
    aps = []
    pos_to_ap = [[] for _ in range(n)]
    ap_id = 0
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(ap)
            for pos in ap:
                pos_to_ap[pos].append(ap_id)
            ap_id += 1
    return aps, pos_to_ap

def _thue_morse_init():
    """Generate Thue-Morse coloring for N positions."""
    colors = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)
    return colors

def _random_init():
    """Generate random coloring."""
    return np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)

def _flip_delta(colors, pos, aps, pos_to_ap):
    """Compute the change in mono-4AP count if we flip position 'pos'.
    Returns delta (negative is improvement)."""
    # For each AP containing pos, check if it's currently monochromatic
    # and what it would be after flip
    current_mono = 0
    after_mono = 0
    new_color = 1 - colors[pos]
    for ap_idx in pos_to_ap[pos]:
        a, b, c, d = aps[ap_idx]
        # Current state
        vals = [colors[a], colors[b], colors[c], colors[d]]
        if len(set(vals)) == 1:
            current_mono += 1
        # After flip
        vals2 = list(vals)
        # Find which position in the AP is 'pos'
        if a == pos:
            vals2[0] = new_color
        elif b == pos:
            vals2[1] = new_color
        elif c == pos:
            vals2[2] = new_color
        elif d == pos:
            vals2[3] = new_color
        if len(set(vals2)) == 1:
            after_mono += 1
    return after_mono - current_mono

def _hill_climb(colors, aps, pos_to_ap, max_iters=50000):
    """Greedy hill climbing: flip a position if it reduces mono-4AP count."""
    n = N
    best_count = _count_mono_4aps_vec(colors)
    for _ in range(max_iters):
        improved = False
        # Try all positions in random order
        order = list(range(n))
        random.shuffle(order)
        for pos in order:
            delta = _flip_delta(colors, pos, aps, pos_to_ap)
            if delta < 0:
                colors[pos] = 1 - colors[pos]
                best_count += delta
                improved = True
        if not improved:
            break
    return colors, best_count

def _simulated_annealing(colors, aps, pos_to_ap, start_temp=10.0, end_temp=0.01, 
                          iters_per_temp=500, n_temps=50, max_time=None):
    """Simulated annealing with deadline guard."""
    n = N
    t0 = time.time()
    best_count = _count_mono_4aps_vec(colors)
    best_colors = colors.copy()
    current_count = best_count
    
    for t_idx in range(n_temps):
        temp = start_temp * (end_temp / start_temp) ** (t_idx / n_temps)
        for _ in range(iters_per_temp):
            if max_time is not None and time.time() - t0 > max_time:
                break
            pos = random.randint(0, n - 1)
            delta = _flip_delta(colors, pos, aps, pos_to_ap)
            if delta < 0 or random.random() < np.exp(-delta / temp):
                colors[pos] = 1 - colors[pos]
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
        if max_time is not None and time.time() - t0 > max_time:
            break
    return best_colors, best_count

def construct():
    t_start = time.time()
    
    # Precompute AP structure
    aps, pos_to_ap = _build_ap_structure()
    
    # Start with Thue-Morse initialization
    colors = _thue_morse_init()
    best_count = _count_mono_4aps_vec(colors)
    best_colors = colors.copy()
    
    # Phase 1: Greedy hill climbing
    colors, count = _hill_climb(colors, aps, pos_to_ap, max_iters=20000)
    if count < best_count:
        best_count = count
        best_colors = colors.copy()
    
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT - 30:
        return [int(c) for c in best_colors]
    
    # Phase 2: Simulated annealing from best state
    colors_sa = best_colors.copy()
    max_time_sa = TIME_LIMIT - 10 - (time.time() - t_start)
    if max_time_sa > 5:
        colors_sa, count_sa = _simulated_annealing(
            colors_sa, aps, pos_to_ap, 
            start_temp=5.0, end_temp=0.005,
            iters_per_temp=200, n_temps=100,
            max_time=max_time_sa
        )
        if count_sa < best_count:
            best_count = count_sa
            best_colors = colors_sa.copy()
    
    # Phase 3: Final hill climb polish
    remaining = TIME_LIMIT - 5 - (time.time() - t_start)
    if remaining > 2:
        colors_final = best_colors.copy()
        colors_final, count_final = _hill_climb(colors_final, aps, pos_to_ap, max_iters=10000)
        if count_final < best_count:
            best_count = count_final
            best_colors = colors_final.copy()
    
    # Verify
    final_count = _count_mono_4aps_vec(best_colors)
    if final_count != best_count:
        # Safety: use verified count
        pass
    
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
