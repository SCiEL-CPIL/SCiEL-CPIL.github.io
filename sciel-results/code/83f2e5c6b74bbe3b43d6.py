import time
import random
import numpy as np

N = 256
DEADLINE = 140.0  # safety margin before 150s kill

def _build_aps():
    """Build list of all 4-term arithmetic progressions (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_position_map(aps):
    """For each position, list of (ap_index, other_positions) it participates in."""
    pos_map = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        for pos in (a, b, c, d):
            others = [x for x in (a, b, c, d) if x != pos]
            pos_map[pos].append((idx, tuple(others)))
    return pos_map

def _count_mono(aps, colors):
    """Count monochromatic 4-APs."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search(aps, pos_map, colors, max_iter=20000, deadline=None):
    """Greedy local search: flip bits that reduce mono-AP count."""
    colors = list(colors)
    n = len(colors)
    num_aps = len(aps)
    
    # Track which APs are currently mono
    # For efficiency, track mono status per AP
    ap_mono = np.zeros(num_aps, dtype=bool)
    for idx, (a, b, c, d) in enumerate(aps):
        if colors[a] == colors[b] == colors[c] == colors[d]:
            ap_mono[idx] = True
    current_count = ap_mono.sum()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        best_gain = 0
        best_pos = -1
        best_new_val = -1
        
        for i in range(n):
            # Compute gain from flipping position i
            # For each AP containing i, check if flipping changes mono status
            gain = 0
            for (ap_idx, others) in pos_map[i]:
                old_val = colors[i]
                new_val = 1 - old_val
                # Check if AP is mono before and after
                old_mono = (old_val == colors[others[0]] == colors[others[1]] == colors[others[2]])
                new_mono = (new_val == colors[others[0]] == colors[others[1]] == colors[others[2]])
                if old_mono and not new_mono:
                    gain += 1
                elif not old_mono and new_mono:
                    gain -= 1
            if gain > best_gain:
                best_gain = gain
                best_pos = i
                best_new_val = 1 - colors[i]
        
        if best_gain > 0 and best_pos >= 0:
            colors[best_pos] = best_new_val
            # Update ap_mono for affected APs
            for (ap_idx, others) in pos_map[best_pos]:
                a, b, c, d = aps[ap_idx]
                if colors[a] == colors[b] == colors[c] == colors[d]:
                    if not ap_mono[ap_idx]:
                        ap_mono[ap_idx] = True
                        current_count += 1
                else:
                    if ap_mono[ap_idx]:
                        ap_mono[ap_idx] = False
                        current_count -= 1
            improved = True
    
    return colors, current_count

def _lfsr_init(seed):
    """Generate a coloring using LFSR-like structure."""
    rng = random.Random(seed)
    colors = []
    for i in range(N):
        # Use a mix of bit-level structure
        # Bit-reversal based pattern with some randomness
        b = i
        # XOR-fold the bits
        x = 0
        for bit in range(8):
            if (i >> bit) & 1:
                x ^= (1 << ((bit * 3 + seed) % 8))
        colors.append(x & 1)
    return colors

def _structured_init(seed):
    """Various structured initializations."""
    rng = random.Random(seed)
    method = seed % 5
    if method == 0:
        # Alternating blocks of size 2
        return [(i // 2) % 2 for i in range(N)]
    elif method == 1:
        # Alternating blocks of size 3
        return [(i // 3) % 2 for i in range(N)]
    elif method == 2:
        # Based on highest set bit
        return [bin(i).count('1') % 2 for i in range(N)]
    elif method == 3:
        # Random
        return [rng.randint(0, 1) for _ in range(N)]
    else:
        # LFSR-like
        return _lfsr_init(seed)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    aps = _build_aps()
    pos_map = _build_position_map(aps)
    num_aps = len(aps)
    
    best_colors = None
    best_count = float('inf')
    
    # Try multiple initializations with local search
    seeds = list(range(20))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        colors = _structured_init(seed)
        colors, count = _local_search(aps, pos_map, colors, max_iter=10000, deadline=deadline - 2)
        
        if count < best_count:
            best_count = count
            best_colors = list(colors)
    
    # If we have time, do more restarts
    if time.time() < deadline - 10 and best_colors is not None:
        for extra_seed in range(20, 50):
            if time.time() > deadline - 3:
                break
            colors = _structured_init(extra_seed)
            colors, count = _local_search(aps, pos_map, colors, max_iter=5000, deadline=deadline - 2)
            if count < best_count:
                best_count = count
                best_colors = list(colors)
    
    if best_colors is None:
        best_colors = _structured_init(0)
    
    return best_colors
