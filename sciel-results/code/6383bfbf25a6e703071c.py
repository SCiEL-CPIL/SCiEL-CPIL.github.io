import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    # Leave 5 seconds for overhead/exit
    deadline = start_time + 55.0
    
    # Correct Thue-Morse generation
    s = [1] * n
    for i in range(1, n):
        if i % 2 == 0:
            s[i] = s[i // 2]
        else:
            s[i] = -s[i // 2]
    
    # Precompute all APs: for each AP, store (start, step, indices)
    # Group APs by which positions they contain for incremental updates
    # ap_by_pos[j] = list of AP indices that contain position j
    # For each AP, we need to track its current sum
    
    # Generate all APs
    all_aps = []  # list of (start, step, [indices])
    for d in range(1, n + 1):
        for a in range(n):
            indices = []
            idx = a
            while idx < n:
                indices.append(idx)
                idx += d
            if len(indices) > 0:
                all_aps.append((a, d, indices))
    
    num_aps = len(all_aps)
    
    # For each position, which APs contain it?
    # ap_by_pos[j] = list of (ap_index, sign_in_ap) where sign is +1 since we just add seq[j]
    ap_by_pos = [[] for _ in range(n)]
    for ap_idx, (a, d, indices) in enumerate(all_aps):
        for j in indices:
            ap_by_pos[j].append(ap_idx)
    
    # Current sum for each AP
    ap_sums = [0] * num_aps
    for ap_idx, (a, d, indices) in enumerate(all_aps):
        for j in indices:
            ap_sums[ap_idx] += s[j]
    
    # Build histogram of |ap_sums| to quickly find max
    # max possible |sum| is n (all +1 or all -1)
    max_possible = n
    # hist[k] = count of APs with |sum| == k
    hist = [0] * (max_possible + 1)
    for sm in ap_sums:
        hist[abs(sm)] += 1
    
    # Current max discrepancy
    current_disc = 0
    for k in range(max_possible, -1, -1):
        if hist[k] > 0:
            current_disc = k
            break
    
    best_s = s[:]
    best_disc = current_disc
    
    rng = random.Random(123)
    
    # Simulated Annealing with incremental updates
    T = 3.0
    T_min = 0.05
    cooling = 0.9995
    
    def get_max_from_hist():
        for k in range(max_possible, -1, -1):
            if hist[k] > 0:
                return k
        return 0
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        old_val = s[i]
        new_val = -old_val
        s[i] = new_val
        
        # Update all APs containing position i
        for ap_idx in ap_by_pos[i]:
            old_sum = ap_sums[ap_idx]
            new_sum = old_sum + (new_val - old_val)  # = old_sum + 2*new_val since new_val = -old_val
            # Update histogram
            hist[abs(old_sum)] -= 1
            ap_sums[ap_idx] = new_sum
            hist[abs(new_sum)] += 1
        
        # Get new max
        new_disc = get_max_from_hist()
        
        delta = new_disc - current_disc
        
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            current_disc = new_disc
            if current_disc < best_disc:
                best_s = s[:]
                best_disc = current_disc
        else:
            # Undo the flip
            s[i] = old_val
            for ap_idx in ap_by_pos[i]:
                old_sum = ap_sums[ap_idx]
                new_sum = old_sum + (old_val - new_val)  # revert
                hist[abs(old_sum)] -= 1
                ap_sums[ap_idx] = new_sum
                hist[abs(new_sum)] += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_s
