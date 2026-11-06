"""Rewritten program: vectorized simulated annealing for 4-AP minimization."""
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115  # safety margin before 150s kill
    
    # Precompute all 4-term arithmetic progressions as index tuples (0-based)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, which APs does it participate in?
    # pos_aps[i] = list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Convert to arrays for fast access
    pos_ap_indices = []
    pos_ap_offsets = []
    for i in range(n):
        if pos_aps[i]:
            indices, offsets = zip(*pos_aps[i])
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int8))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int8))
    
    def count_mono_4ap(color):
        """Count monochromatic 4-APs. color is 0/1 array of length n."""
        # For each AP, check if all 4 positions have same color
        c = color[aps_arr]  # shape (num_aps, 4)
        # All same means all equal to first
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))
    
    def delta_flip(color, pos):
        """Compute change in mono-4AP count if we flip position 'pos'."""
        # Get all APs involving this position
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        
        # Current colors of all 4 positions in these APs
        sub = color[aps_arr[ap_idx]]  # shape (len, 4)
        off = pos_ap_offsets[pos]  # shape (len,)
        
        # Current mono status
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        
        # After flip: the color at 'off' position changes
        new_sub = sub.copy()
        new_sub[np.arange(len(ap_idx)), off] = 1 - new_sub[np.arange(len(ap_idx)), off]
        new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
        new_count = int(np.sum(new_mono))
        
        return new_count - curr_count
    
    # Initialize with Thue-Morse sequence (known good starting point)
    rng = np.random.default_rng(42)
    color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    best_color = color.copy()
    best_score = count_mono_4ap(color)
    current_score = best_score
    
    # Simulated annealing
    T_start = 2.0
    T_end = 0.01
    num_iterations = 200000
    
    # Precompute log space for temperature schedule
    # Use geometric cooling
    cool_rate = (T_end / T_start) ** (1.0 / num_iterations)
    T = T_start
    
    elapsed_start = time.time()
    
    for it in range(num_iterations):
        # Check deadline
        if time.time() > deadline - 5:
            break
        
        # Pick random position
        pos = rng.integers(0, n)
        
        # Compute delta
        delta = delta_flip(color, pos)
        
        # Accept if improving or with probability exp(-delta/T)
        if delta <= 0:
            color[pos] = 1 - color[pos]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_color = color.copy()
        else:
            # Metropolis criterion
            if rng.random() < np.exp(-delta / T):
                color[pos] = 1 - color[pos]
                current_score += delta
        
        # Cool down
        T *= cool_rate
    
    # Final local search refinement on best solution
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for pos in range(n):
            if time.time() > deadline - 10:
                break
            delta = delta_flip(best_color, pos)
            if delta < 0:
                best_color[pos] = 1 - best_color[pos]
                best_score += delta
                improved = True
    
    return [int(c) for c in best_color]
