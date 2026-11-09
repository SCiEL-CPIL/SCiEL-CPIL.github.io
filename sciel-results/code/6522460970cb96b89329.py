"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _build_ap_list():
    """Build list of all 4-term APs (a, a+d, a+2d, a+3d) with d>=1 in 0..255."""
    aps = []
    n = 256
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono_4aps(colors):
    """Count monochromatic 4-term APs. colors is a numpy array of 0/1, length 256."""
    n = len(colors)
    count = 0
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            c0 = colors[a]
            if c0 == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 256
    aps = _build_ap_list()
    n_aps = len(aps)
    
    # Precompute AP indices as numpy array for vectorized operations
    ap_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    # Thue-Morse as initial coloring
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    best_colors = colors.copy()
    best_count = _count_mono_4aps(best_colors)
    
    # Local search: flip individual bits to reduce mono-4-AP count
    # For each position, compute delta if we flip it
    
    def flip_delta(colors, pos):
        """Compute change in mono-4-AP count if we flip position pos."""
        # Find all APs containing pos
        # For each such AP, check if flipping changes mono status
        delta = 0
        for ap in aps:
            if pos not in ap:
                continue
            # Check current mono status
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            was_mono = (c0 == c1 == c2 == c3)
            
            # After flip
            new_colors = list(colors)
            new_colors[pos] = 1 - colors[pos]
            nc0, nc1, nc2, nc3 = (new_colors[ap[0]], new_colors[ap[1]], 
                                   new_colors[ap[2]], new_colors[ap[3]])
            is_mono_now = (nc0 == nc1 == nc2 == nc3)
            
            if is_mono_now and not was_mono:
                delta += 1
            elif was_mono and not is_mono_now:
                delta -= 1
        
        return delta
    
    # Simulated annealing / greedy local search
    rng = np.random.default_rng(42)
    
    # Precompute for each position which APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Convert to numpy arrays for speed
    pos_aps_arr = [np.array(pos_to_aps[i], dtype=np.int32) for i in range(n)]
    
    # Efficient delta computation
    def compute_flip_delta_fast(colors, pos):
        """Fast delta computation using precomputed AP membership."""
        delta = 0
        for ap_idx in pos_aps_arr[pos]:
            ap = ap_arr[ap_idx]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            was_mono = (c0 == c1 == c2 == c3)
            
            # Determine new values
            nc = [c0, c1, c2, c3]
            for j, p in enumerate(ap):
                if p == pos:
                    nc[j] = 1 - nc[j]
            
            is_mono_now = (nc[0] == nc[1] == nc[2] == nc[3])
            
            if is_mono_now and not was_mono:
                delta += 1
            elif was_mono and not is_mono_now:
                delta -= 1
        
        return delta
    
    # Greedy + simulated annealing
    current = colors.copy()
    current_count = best_count
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    # Multi-start with different initial colorings
    initial_colorings = [best_colors.copy()]
    # Random colorings
    for _ in range(3):
        rc = rng.integers(0, 2, size=n).astype(np.int8)
        initial_colorings.append(rc)
    # Block patterns
    for block in [2, 4, 8, 16]:
        bc = np.array([(i // block) % 2 for i in range(n)], dtype=np.int8)
        initial_colorings.append(bc)
    
    for init_idx, init_colors in enumerate(initial_colorings):
        if time.time() > deadline:
            break
        
        current = init_colors.copy()
        current_count = _count_mono_4aps(current)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = current.copy()
        
        T = 10.0
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline and T > min_T:
            # Pick random position
            pos = rng.integers(0, n)
            
            delta = compute_flip_delta_fast(current, pos)
            
            if delta < 0 or (delta == 0 and rng.random() < 0.1):
                # Accept
                current[pos] = 1 - current[pos]
                current_count += delta
                no_improve = 0
            elif delta > 0 and rng.random() < np.exp(-delta / T):
                # Accept worse with probability
                current[pos] = 1 - current[pos]
                current_count += delta
                no_improve += 1
            else:
                no_improve += 1
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current.copy()
            
            T *= cooling
            
            if no_improve > max_no_improve:
                # Restart with current T
                no_improve = 0
                T = max(T * 1.5, min_T)
    
    return list(best_colors)
# EVOLVE-BLOCK-END
