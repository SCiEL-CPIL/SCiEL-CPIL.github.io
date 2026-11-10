import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    max_time = 55.0  # Leave 5s buffer
    
    # 1. Pre-compute all Arithmetic Progressions
    # An AP is (start, step, length) with length >= 1.
    # Indices are 0-based.
    # AP is valid if start + (length-1)*step < n.
    
    ap_indices = []
    # We store the list of indices for each AP to compute sums easily?
    # No, storing lists of indices for 200k APs is memory heavy but okay.
    # Better: store (start, step, length) and compute sum on the fly?
    # Or store the indices as a numpy array for each AP?
    
    # Let's generate (start, step, length) tuples
    aps = []
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        for start in range(n):
            # Max length for this start and step
            # start + (k-1)d < n  => k-1 < (n-start)/d => k <= (n-start-1)//d + 1
            # But we also need start >= 0.
            # Actually, for a fixed d and start, the APs are prefixes of the sequence start, start+d, ...
            # So for each (start, d), there are L APs of length 1, 2, ..., L_max.
            # This is redundant. An AP of length 2 is a subsequence of length 3?
            # The task says "ANY arithmetic progression".
            # Does "progression" imply contiguous in the AP sense?
            # "contained in [1,256] ... any start, positive step and length".
            # Yes, usually this means the set $\{a, a+d, ..., a+(k-1)d\}$.
            # So for a fixed (a, d), the APs are $\{a\}$, $\{a, a+d\}$, $\{a, a+d, a+2d\}$, etc.
            # So we can just iterate a, d, and k.
            
            max_k = (n - 1 - start) // d + 1
            if max_k < 1:
                continue
            for k in range(1, max_k + 1):
                # Indices: start, start+d, ..., start+(k-1)d
                aps.append((start, d, k))
                
    num_aps = len(aps)
    
    # Pre-compute for each sign index i, which APs contain it?
    # This is needed for efficient updates.
    # But first, let's just compute the sums directly from signs for initialization.
    
    # To make it fast, let's represent each AP as a numpy array of indices?
    # That's too much memory.
    # Let's just store (start, d, k).
    
    # Initialize random signs
    signs = np.random.choice([-1, 1], size=n).astype(np.int8)
    
    # Function to compute all AP sums
    def compute_sums(s):
        sums = np.zeros(num_aps, dtype=np.int8) # Sums are small, int8 or int16 is enough. Max sum 256.
        for idx, (a, d, k) in enumerate(aps):
            # Sum s[a : a+k*d : d]
            # This is a bit slow in Python loop.
            # Let's use numpy slicing.
            # s[a::d] gives the full sequence. We take first k.
            # But slicing creates a view/copy.
            # 200k iterations * slice cost might be slow.
            # Let's try to optimize this.
            pass
        return sums

    # Better approach for compute_sums:
    # For each d, group APs.
    # Actually, let's just use a loop but with numpy sum on slices.
    
    def get_all_sums(s):
        sums = np.empty(num_aps, dtype=np.int16)
        for idx, (a, d, k) in enumerate(aps):
            # Extract slice
            # s[a : a + k*d : d]
            # Note: a + k*d might exceed n, but slice handles it.
            # However, we generated k such that a+(k-1)d < n, so a+k*d <= n + d?
            # No, a+(k-1)d < n.
            # So s[a : a+k*d : d] is safe.
            sums[idx] = np.sum(s[a : a + k * d : d])
        return sums

    # Initial sums
    sums = get_all_sums(signs)
    max_val = np.max(np.abs(sums))
    
    # Pre-compute "which APs contain index i"
    # ap_containing[i] = list of ap indices
    ap_containing = [[] for _ in range(n)]
    for idx, (a, d, k) in enumerate(aps):
        for j in range(k):
            pos = a + j * d
            ap_containing[pos].append(idx)
            
    # Local Search
    current_max = max_val
    
    while time.time() - start_time < max_time:
        # Find the AP with the maximum absolute sum
        # We need to track which AP is the worst.
        # Since sums change, we need to re-find the max or track it.
        # Re-finding max is O(num_aps). 200k is fast.
        max_abs_idx = np.argmax(np.abs(sums))
        current_max = np.abs(sums[max_abs_idx])
        
        a, d, k = aps[max_abs_idx]
        ap_elements = [a + j * d for j in range(k)]
        
        best_flip = None
        best_new_max = current_max
        
        # Try flipping each element in the worst AP
        for pos in ap_elements:
            # Flip pos
            signs[pos] *= -1
            
            # Update sums for all APs containing pos
            # We need to update sums array.
            # Old sum for AP j containing pos: sums[j]
            # New sum: sums[j] - 2 * signs[pos] (since we flipped, diff is -2*old_sign)
            # Wait, signs[pos] is already flipped.
            # Let old_sign = signs[pos] (which is now -1 or 1)
            # Actually, let's save old sign.
            
            # Re-do:
            # We flipped signs[pos].
            # For each AP j in ap_containing[pos]:
            #   sums[j] -= 2 * old_sign  (where old_sign was the value BEFORE flip)
            #   But we already flipped it.
            #   So sums[j] += 2 * new_sign? No.
            #   Sum = ... + old_sign + ...
            #   New Sum = ... + new_sign + ... = Old Sum - old_sign + new_sign
            #   new_sign = -old_sign
            #   New Sum = Old Sum - 2 * old_sign
            #   We don't have old_sign anymore.
            #   But new_sign is known. old_sign = -new_sign.
            #   New Sum = Old Sum + 2 * new_sign
            
            new_sign = signs[pos]
            
            for ap_idx in ap_containing[pos]:
                sums[ap_idx] += 2 * new_sign
                
            # Find new global max
            # Optimization: Only the updated sums changed.
            # But finding the global max requires scanning all.
            # We can keep a histogram or just scan.
            # Scanning 200k int16s is ~0.1ms?
            new_max = np.max(np.abs(sums))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip = pos
            else:
                # Revert
                signs[pos] *= -1
                for ap_idx in ap_containing[pos]:
                    sums[ap_idx] -= 2 * new_sign # Revert the addition
                # Note: new_sign was the sign after flip.
                # Reverting flip means going back to -new_sign.
                # So we subtract 2 * new_sign from the sums that were increased by 2 * new_sign.
                # Correct.
                
        if best_flip is not None:
            # Apply the best flip
            pos = best_flip
            signs[pos] *= -1
            new_sign = signs[pos]
            for ap_idx in ap_containing[pos]:
                sums[ap_idx] += 2 * new_sign
            current_max = best_new_max
        else:
            # No improvement found in the worst AP.
            # Try a random flip to escape local minimum (Simulated Annealing / Random Walk)
            # Pick a random position
            pos = np.random.randint(n)
            old_sign = signs[pos]
            signs[pos] *= -1
            new_sign = signs[pos]
            for ap_idx in ap_containing[pos]:
                sums[ap_idx] += 2 * new_sign
            
            new_max = np.max(np.abs(sums))
            
            # Accept if better or equal? Or always accept to move?
            # If we always accept, we might drift away.
            # Let's use a simple criterion: Accept if new_max <= current_max * 1.1 (tolerance)
            # Or just accept if it doesn't get too much worse.
            if new_max > current_max + 5:
                # Revert
                signs[pos] *= -1
                for ap_idx in ap_containing[pos]:
                    sums[ap_idx] -= 2 * new_sign
            else:
                current_max = new_max
                
    return signs.tolist()
