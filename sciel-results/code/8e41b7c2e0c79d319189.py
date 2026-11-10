import time
import numpy as np
import random

_N = 256
_DEADLINE = 85.0

def _generate_aps():
    """
    Generate all arithmetic progressions in [0, N-1].
    Returns:
        ap_starts: np.array of start indices
        ap_steps: np.array of steps
        ap_lens: np.array of lengths
        aps_containing: list of lists, aps_containing[i] = list of AP IDs containing index i
    """
    ap_starts = []
    ap_steps = []
    ap_lens = []
    aps_containing = [[] for _ in range(_N)]
    
    ap_id = 0
    for d in range(1, _N):
        for r in range(d):
            # Sequence: r, r+d, r+2d, ...
            seq = list(range(r, _N, d))
            len_seq = len(seq)
            # All contiguous subarrays of seq are APs with step d
            for i in range(len_seq):
                for j in range(i, len_seq):
                    start_idx = seq[i]
                    length = j - i + 1
                    # Store AP
                    ap_starts.append(start_idx)
                    ap_steps.append(d)
                    ap_lens.append(length)
                    
                    # Add to containing lists
                    for k in range(length):
                        pos = start_idx + k * d
                        aps_containing[pos].append(ap_id)
                    
                    ap_id += 1
                    
    return (np.array(ap_starts, dtype=np.int32), 
            np.array(ap_steps, dtype=np.int32), 
            np.array(ap_lens, dtype=np.int32), 
            aps_containing)

def _greedy_construct():
    s = np.zeros(_N, dtype=np.int8)
    for i in range(_N):
        best_sign = 1
        best_disc = float('inf')
        for sign in (1, -1):
            s[i] = sign
            # Fast check: just check APs ending at i?
            # No, we need global max for the prefix.
            # But for greedy, we can just use the incremental structure if we build it up?
            # Simpler: just use the O(N^2) check for the prefix.
            # It's only done 256 times.
            max_disc = 1
            for d in range(1, i+1):
                for r in range(d):
                    if r > i: continue
                    seq = s[r::d]
                    # Only consider elements <= i
                    # Since s is 0 after i, this is tricky.
                    # Better to pass a slice.
                    seq_slice = s[r:i+1:d]
                    if len(seq_slice) <= 1: continue
                    csum = np.cumsum(seq_slice)
                    disc = np.max(csum) - np.min(csum)
                    if disc > max_disc:
                        max_disc = disc
            if max_disc < best_disc:
                best_disc = max_disc
                best_sign = sign
        s[i] = best_sign
    return s

def construct():
    start_time = time.time()
    
    # 1. Generate APs
    ap_starts, ap_steps, ap_lens, aps_containing = _generate_aps()
    num_aps = len(ap_starts)
    
    # 2. Greedy Initialization
    s = _greedy_construct()
    
    # 3. Initialize Sums and Counts
    # ap_sums[i] is the sum of AP i
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    # Calculate initial sums
    for k in range(num_aps):
        start = ap_starts[k]
        step = ap_steps[k]
        length = ap_lens[k]
        # Sum s[start : start + length*step : step]
        ap_sums[k] = np.sum(s[start : start + length*step : step])
        
    # Build sum_counts
    # Max possible sum is N=256. Min is -256.
    # Range [-256, 256]. Size 513.
    OFFSET = _N
    sum_counts = np.zeros(2 * _N + 1, dtype=np.int32)
    
    for sum_val in ap_sums:
        sum_counts[sum_val + OFFSET] += 1
        
    # Find initial max abs sum
    max_abs_sum = 0
    for val in range(_N, -1, -1):
        if sum_counts[val + OFFSET] > 0 or sum_counts[-val + OFFSET] > 0:
            max_abs_sum = val
            break
            
    best_s = s.copy()
    best_disc = max_abs_sum
    
    # 4. Local Search
    rng = random.Random(42)
    
    while time.time() - start_time < _DEADLINE:
        idx = rng.randint(0, _N - 1)
        
        old_val = s[idx]
        new_val = -old_val
        delta = new_val - old_val # +/- 2
        
        # Update AP sums
        affected_aps = aps_containing[idx]
        
        # Store old sums to revert if needed
        old_aps_sums = []
        
        for ap_id in affected_aps:
            old_sum = ap_sums[ap_id]
            new_sum = old_sum + delta
            
            # Update counts
            sum_counts[old_sum + OFFSET] -= 1
            sum_counts[new_sum + OFFSET] += 1
            
            # Update ap_sums
            ap_sums[ap_id] = new_sum
            
            # Track if we need to update max_abs_sum
            old_aps_sums.append(old_sum)
            
            # Check if new_sum increases max_abs_sum
            if abs(new_sum) > max_abs_sum:
                max_abs_sum = abs(new_sum)
                
        # If max_abs_sum decreased, we need to find the new max
        # This happens if the AP that had the max sum was modified and its sum decreased
        # We can check if sum_counts[max_abs_sum + OFFSET] + sum_counts[-max_abs_sum + OFFSET] == 0
        # If so, decrement max_abs_sum until we find a non-zero count.
        
        # Optimization: Only do the expensive search if the max AP was among the affected ones?
        # Or just always check.
        # Since max_abs_sum is small (likely < 10), the loop is very fast.
        
        # However, we only need to decrease max_abs_sum if the count at the current max is 0.
        # It might be that max_abs_sum stayed the same or increased.
        
        # Let's check if the current max_abs_sum is still valid.
        # If sum_counts[max_abs_sum + OFFSET] == 0 and sum_counts[-max_abs_sum + OFFSET] == 0:
        #     Then we need to find the new max.
        
        # But wait, if we flipped a sign, the max could only increase or decrease.
        # If it increased, we updated it above.
        # If it decreased, it means the AP(s) that held the max were affected and their sums dropped.
        
        # Let's just verify the current max_abs_sum.
        if sum_counts[max_abs_sum + OFFSET] == 0 and (max_abs_sum == 0 or sum_counts[-max_abs_sum + OFFSET] == 0):
            # Need to find new max
            while max_abs_sum > 0 and sum_counts[max_abs_sum + OFFSET] == 0 and sum_counts[-max_abs_sum + OFFSET] == 0:
                max_abs_sum -= 1
                
        # Accept or Revert
        if max_abs_sum <= best_disc:
            # Accept
            s[idx] = new_val
            best_disc = max_abs_sum
            if max_abs_sum < best_disc: # This logic is redundant, best_disc is updated
                pass
            best_s = s.copy()
        else:
            # Revert
            s[idx] = old_val
            for ap_id, old_sum in zip(affected_aps, old_aps_sums):
                new_sum = ap_sums[ap_id]
                sum_counts[new_sum + OFFSET] -= 1
                sum_counts[old_sum + OFFSET] += 1
                ap_sums[ap_id] = old_sum
                
            # Revert max_abs_sum
            # We need to restore the max_abs_sum to what it was before the flip.
            # We didn't store the old max_abs_sum.
            # But we can recompute it or store it.
            # Let's store old_max_abs_sum before the flip.
            
            # Correction: I need to store old_max_abs_sum.
            
    return best_s.tolist()

# Refactored construct with proper state saving
def construct():
    start_time = time.time()
    
    ap_starts, ap_steps, ap_lens, aps_containing = _generate_aps()
    num_aps = len(ap_starts)
    
    s = _greedy_construct()
    
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for k in range(num_aps):
        start = ap_starts[k]
        step = ap_steps[k]
        length = ap_lens[k]
        ap_sums[k] = np.sum(s[start : start + length*step : step])
        
    OFFSET = _N
    sum_counts = np.zeros(2 * _N + 1, dtype=np.int32)
    for sum_val in ap_sums:
        sum_counts[sum_val + OFFSET] += 1
        
    max_abs_sum = 0
    for val in range(_N, -1, -1):
        if sum_counts[val + OFFSET] > 0 or sum_counts[-val + OFFSET] > 0:
            max_abs_sum = val
            break
            
    best_s = s.copy()
    best_disc = max_abs_sum
    
    rng = random.Random(42)
    
    while time.time() - start_time < _DEADLINE:
        idx = rng.randint(0, _N - 1)
        
        old_val = s[idx]
        new_val = -old_val
        delta = new_val - old_val
        
        affected_aps = aps_containing[idx]
        old_aps_sums = [ap_sums[ap_id] for ap_id in affected_aps]
        old_max_abs_sum = max_abs_sum
        
        for ap_id in affected_aps:
            old_sum = ap_sums[ap_id]
            new_sum = old_sum + delta
            
            sum_counts[old_sum + OFFSET] -= 1
            sum_counts[new_sum + OFFSET] += 1
            ap_sums[ap_id] = new_sum
            
            if abs(new_sum) > max_abs_sum:
                max_abs_sum = abs(new_sum)
                
        if sum_counts[max_abs_sum + OFFSET] == 0 and (max_abs_sum == 0 or sum_counts[-max_abs_sum + OFFSET] == 0):
            while max_abs_sum > 0 and sum_counts[max_abs_sum + OFFSET] == 0 and sum_counts[-max_abs_sum + OFFSET] == 0:
                max_abs_sum -= 1
                
        if max_abs_sum <= best_disc:
            s[idx] = new_val
            best_disc = max_abs_sum
            best_s = s.copy()
        else:
            s[idx] = old_val
            for ap_id, old_sum in zip(affected_aps, old_aps_sums):
                new_sum = ap_sums[ap_id]
                sum_counts[new_sum + OFFSET] -= 1
                sum_counts[old_sum + OFFSET] += 1
                ap_sums[ap_id] = old_sum
            max_abs_sum = old_max_abs_sum

    return best_s.tolist()
