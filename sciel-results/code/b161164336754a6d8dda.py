import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Generate all APs within [0, n)
    # We only care about APs with length >= 2 for the max, 
    # because singletons always have sum 1. 
    # If we achieve max 1, we are optimal.
    ap_indices = []
    for d in range(1, n):
        for a in range(d):
            seq = list(range(a, n, d))
            if len(seq) >= 2:
                ap_indices.append(np.array(seq, dtype=np.int32))
                
    m = len(ap_indices)
    
    # Map index to list of AP indices
    index_to_aps = [[] for _ in range(n)]
    for ap_id, ap in enumerate(ap_indices):
        for idx in ap:
            index_to_aps[idx].append(ap_id)
            
    # Convert to numpy arrays for faster indexing in loops if needed, 
    # but list of lists is fine for small sizes.
    # Actually, let's keep them as lists for flexibility, 
    # but we access them frequently.
    
    def evaluate_initial(signs):
        ap_sums = np.zeros(m, dtype=np.int16)
        for ap_id, ap in enumerate(ap_indices):
            ap_sums[ap_id] = np.sum(signs[ap])
        
        # Build histogram
        hist = np.zeros(n + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        # Find max
        current_max = 0
        for val in range(n, -1, -1):
            if hist[val] > 0:
                current_max = val
                break
                
        return ap_sums, hist, current_max

    def local_search(signs, ap_sums, hist, current_max, time_limit):
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < time_limit:
            if no_improve > max_no_improve:
                break
                
            if current_max <= 1:
                break
                
            # Find APs with max sum
            # We need to find one AP with |sum| == current_max
            # We can iterate ap_sums, but that's O(M).
            # Alternatively, we can pick a random index and check if it's in a critical AP?
            # Better: Precompute list of APs for each possible sum magnitude?
            # That's expensive to maintain.
            
            # Let's just pick a random AP from the critical set.
            # To avoid O(M) scan for critical APs, we can just pick a random index 
            # and see if flipping it helps? No, that's random walk.
            
            # Strategy: Pick a random AP. If it's critical, flip a random element in it.
            # If not, maybe skip?
            # Or: Maintain a list of critical APs?
            # Let's do a simple check: 
            # Pick a random AP id.
            # If abs(ap_sums[ap_id]) == current_max, pick a random index in it and flip.
            # Else, try a few more random APs?
            # If we miss too often, we are inefficient.
            
            # Better: 
            # We know current_max. 
            # We can find the first AP with abs(sum)==current_max by scanning.
            # M is ~16000. Scanning is fast.
            
            critical_ap_id = -1
            for i in range(m):
                if abs(ap_sums[i]) == current_max:
                    critical_ap_id = i
                    break
            
            if critical_ap_id == -1:
                # Should not happen
                break
                
            # Pick a random index from this critical AP
            ap = ap_indices[critical_ap_id]
            flip_idx = ap[random.randint(0, len(ap)-1)]
            
            # Perform flip
            old_sign = signs[flip_idx]
            new_sign = -old_sign
            signs[flip_idx] = new_sign
            
            # Update histogram and sums
            affected_aps = index_to_aps[flip_idx]
            delta = 2 * new_sign
            
            for ap_id in affected_aps:
                old_val = ap_sums[ap_id]
                new_val = old_val + delta
                hist[abs(old_val)] -= 1
                hist[abs(new_val)] += 1
                ap_sums[ap_id] = new_val
                
            # Find new max
            new_max = 0
            for val in range(n, -1, -1):
                if hist[val] > 0:
                    new_max = val
                    break
                    
            if new_max < current_max:
                current_max = new_max
                no_improve = 0
            else:
                # Revert
                signs[flip_idx] = old_sign
                for ap_id in affected_aps:
                    new_val = ap_sums[ap_id]
                    old_val = new_val - delta
                    hist[abs(new_val)] -= 1
                    hist[abs(old_val)] += 1
                    ap_sums[ap_id] = old_val
                no_improve += 1
                
        return signs, ap_sums, hist, current_max

    best_signs = None
    best_max = float('inf')
    
    # Try multiple initializations
    num_attempts = 10
    
    for k in range(num_attempts):
        if time.time() > deadline - 5:
            break
            
        # Determine time for this attempt
        time_left = deadline - time.time()
        time_per_attempt = time_left / (num_attempts - k)
        time_limit = time.time() + time_per_attempt
        
        if k == 0:
            # Thue-Morse
            signs = np.zeros(n, dtype=np.int8)
            for i in range(n):
                signs[i] = 1 if (bin(i).count('1') % 2 == 0) else -1
        elif k == 1:
            # Random
            signs = np.random.choice([-1, 1], size=n).astype(np.int8)
        else:
            # Perturbed best
            if best_signs is not None:
                signs = best_signs.copy()
                # Flip 5 random bits
                for _ in range(5):
                    idx = random.randint(0, n-1)
                    signs[idx] = -signs[idx]
            else:
                signs = np.random.choice([-1, 1], size=n).astype(np.int8)
                
        ap_sums, hist, current_max = evaluate_initial(signs)
        
        signs, ap_sums, hist, current_max = local_search(signs, ap_sums, hist, current_max, time_limit)
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            
            if best_max == 1:
                break
        
    return best_signs.tolist()
