import time
import random
import numpy as np
from collections import Counter

def _build_ap_data(n):
    """
    Pre-compute AP masks, lengths, and reverse index for incremental updates.
    """
    ap_masks = []
    ap_lengths = []
    reverse_index = [[] for _ in range(n)]
    
    for step in range(1, n + 1):
        for start in range(n):
            mask = 0
            length = 0
            idx = start
            while idx < n:
                mask |= (1 << idx)
                length += 1
                idx += step
            
            if length > 0:
                ap_id = len(ap_masks)
                ap_masks.append(mask)
                ap_lengths.append(length)
                
                for j in range(start, n, step):
                    reverse_index[j].append(ap_id)
                    
    return ap_masks, ap_lengths, reverse_index

def _thue_morse_init(n):
    state = 0
    for i in range(n):
        if i.bit_count() % 2 == 1:
            state |= (1 << i)
    return state

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    ap_masks, ap_lengths, reverse_index = _build_ap_data(n)
    num_aps = len(ap_masks)
    
    current_state = _thue_morse_init(n)
    
    ap_sums = [0] * num_aps
    disc_counter = Counter()
    
    for k in range(num_aps):
        c = (current_state & ap_masks[k]).bit_count()
        s = 2 * c - ap_lengths[k]
        ap_sums[k] = s
        disc_counter[abs(s)] += 1
        
    current_max_disc = max(disc_counter.keys()) if disc_counter else 0
    
    T = 1.0
    cooling = 0.9995
    min_T = 1e-6
    rng = random.Random(42)
    
    best_state = current_state
    best_disc = current_max_disc
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        affected_aps = reverse_index[i]
        
        if not affected_aps:
            continue
            
        old_vals = [ap_sums[k] for k in affected_aps]
        
        bit_val = (current_state >> i) & 1
        delta_sum = 2 if bit_val == 0 else -2
        
        # Apply flip
        for k in affected_aps:
            ap_sums[k] += delta_sum
            
        # Update Counter
        for v in old_vals:
            av = abs(v)
            disc_counter[av] -= 1
            if disc_counter[av] == 0:
                del disc_counter[av]
                
        for k in affected_aps:
            av = abs(ap_sums[k])
            disc_counter[av] += 1
            
        new_max_disc = max(disc_counter.keys()) if disc_counter else 0
        delta_obj = new_max_disc - current_max_disc
        
        accepted = False
        if delta_obj <= 0:
            accepted = True
        else:
            prob = np.exp(-delta_obj / T)
            if rng.random() < prob:
                accepted = True
                
        if accepted:
            current_state ^= (1 << i)
            current_max_disc = new_max_disc
            
            if current_max_disc < best_disc:
                best_disc = current_max_disc
                best_state = current_state
        else:
            # Revert
            current_state ^= (1 << i)
            
            for k in affected_aps:
                ap_sums[k] -= delta_sum
                
            # Revert Counter
            # Remove New (which are currently in counter) and Add Old
            for k in affected_aps:
                # Current ap_sums[k] is Old (reverted).
                # New value was ap_sums[k] + delta_sum.
                new_val = ap_sums[k] + delta_sum
                av_new = abs(new_val)
                disc_counter[av_new] -= 1
                if disc_counter[av_new] == 0:
                    del disc_counter[av_new]
                    
            for v in old_vals:
                av_old = abs(v)
                disc_counter[av_old] += 1
                
        T *= cooling
        if T < min_T:
            T = min_T

    result = []
    for i in range(n):
        if (best_state >> i) & 1:
            result.append(1)
        else:
            result.append(-1)
            
    return result

if __name__ == "__main__":
    signs = construct()
    print(len(signs))
    print(signs[:10])
