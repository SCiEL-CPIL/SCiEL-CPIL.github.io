"""Optimized two-colouring of [1,256] minimizing monochromatic 4-APs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
MAX_D = N // 3  # 85

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence: t(i) = popcount(i) % 2."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_indices():
    """Precompute all 4-term APs and for each position, which APs contain it."""
    # For each position p, list of (ap_index, role) where role is 0,1,2,3
    # But more useful: for each position p, list of AP indices that contain p
    # We'll store APs as tuples (a, d) and for each p, which APs it's in
    
    # First, enumerate all APs
    aps = []  # list of (a, d, positions)
    ap_index = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            positions = [a - 1, a - 1 + d, a - 1 + 2*d, a - 1 + 3*d]  # 0-indexed
            aps.append((a, d, positions))
            ap_index += 1
    
    total_aps = len(aps)
    
    # For each position, which APs does it participate in?
    # pos_to_aps[p] = list of (ap_idx, role) where role is 0,1,2,3
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, d, positions) in enumerate(aps):
        for role, p in enumerate(positions):
            pos_to_aps[p].append((idx, role))
    
    return aps, pos_to_aps

# Precompute once at module level
_APS, _POS_TO_APS = _precompute_ap_indices()
_TOTAL_APS = len(_APS)

def _count_mono4_fast(c):
    """Count monochromatic 4-term APs using precomputed structure."""
    count = 0
    for idx, (a, d, positions) in enumerate(_APS):
        p0, p1, p2, p3 = positions
        if c[p0] == c[p1] == c[p2] == c[p3]:
            count += 1
    return count

def _count_mono4_vectorized(c):
    """Vectorized count of monochromatic 4-APs."""
    count = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            continue
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += int(np.sum(mask))
    return count

def _delta_flip(c, pos):
    """Compute change in mono-4-AP count when flipping position pos.
    Returns the delta (new_count - old_count)."""
    old_val = c[pos]
    new_val = 1 - old_val
    
    # For each AP containing pos, check if it was mono before and after
    delta = 0
    for ap_idx, role in _POS_TO_APS[pos]:
        a, d, positions = _APS[ap_idx]
        p0, p1, p2, p3 = positions
        
        # Get colors of all 4 positions
        colors = np.array([c[p0], c[p1], c[p2], c[p3]], dtype=np.int8)
        
        # Check if currently monochromatic
        is_mono_before = (colors[0] == colors[1]) and (colors[1] == colors[2]) and (colors[2] == colors[3])
        
        # Simulate flip
        colors[role] = new_val
        is_mono_after = (colors[0] == colors[1]) and (colors[1] == colors[2]) and (colors[2] == colors[3])
        
        if is_mono_before and not is_mono_after:
            delta -= 1
        elif not is_mono_before and is_mono_after:
            delta += 1
    
    return delta

def _local_search_incremental(c, time_limit):
    """Local search using incremental delta computation."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try positions in random order for diversity
        order = list(range(N))
        random.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    
    return best_c, best_count

def _simulated_annealing(c, time_limit):
    """Simulated annealing to escape local minima."""
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4_vectorized(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling = 0.995
    
    while (time.time() - start) < time_limit and T > T_min:
        # Pick random position
        pos = random.randint(0, N - 1)
        delta = _delta_flip(cur_c, pos)
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            cur_c[pos] = 1 - cur_c[pos]
            cur_count += delta
            if cur_count < best_count:
                best_count = cur_count
                best_c = cur_c.copy()
        
        T *= cooling
    
    return best_c, best_count

def _two_flip_search(c, time_limit):
    """Try flipping pairs of positions simultaneously."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try a limited set of pairs
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
        random.shuffle(pairs)
        
        # Limit pairs tried per iteration
        max_pairs = 500
        for i, j in pairs[:max_pairs]:
            if (time.time() - start) >= time_limit:
                break
            # Flip both
            best_c[i] = 1 - best_c[i]
            best_c[j] = 1 - best_c[j]
            new_count = _count_mono4_vectorized(best_c)
            if new_count < best_count:
                best_count = new_count
                improved = True
            else:
                best_c[i] = 1 - best_c[i]
                best_c[j] = 1 - best_c[j]
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    c = _thue_morse_fast(N)
    best_c = c.copy()
    best_count = _count_mono4_vectorized(best_c)
    
    # Phase 1: Local search with incremental deltas
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 50))
    
    # Phase 2: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 15:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 15, 40))
        if sa_count < best_count:
            best_c = sa_c
            best_count = sa_count
    
    # Phase 3: More local search
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 30))
    
    # Phase 4: Two-flip search
    remaining = deadline - time.time()
    if remaining > 15:
        best_c, best_count = _two_flip_search(best_c, min(remaining - 15, 25))
    
    # Phase 5: Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 20))
    
    # Phase 6: Try different Thue-Morse offsets with local search
    remaining = deadline - time.time()
    if remaining > 15:
        for offset in range(0, 8):
            if time.time() >= deadline - 10:
                break
            c2 = _thue_morse_fast(N)
            c2 = np.roll(c2, offset)
            c2, c2_count = _local_search_incremental(c2, min(8, deadline - time.time() - 5))
            if c2_count < best_count:
                best_c = c2
                best_count = c2_count
    
    # Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 15))
    
    # Convert to list of ints
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END
