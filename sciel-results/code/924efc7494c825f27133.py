import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights
    weights = np.zeros(max_val, dtype=np.int8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    best_code = []
    best_size = 0
    
    # Try different linear code sizes
    for k in [8, 7, 6, 5]:
        if time.time() > deadline - 15:
            break
            
        # 1. Construct Linear Code of size 2^k
        basis = []
        current_lin = [0]
        
        for g in range(1, max_val):
            if len(basis) >= k:
                break
            if time.time() > deadline - 15:
                break
                
            valid = True
            for c in current_lin:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
                    
            if valid:
                basis.append(g)
                old_code = current_lin
                current_lin = []
                for c in old_code:
                    current_lin.append(c)
                    current_lin.append(c ^ g)
                    
        if len(current_lin) < (1 << k):
            continue # Couldn't build full code
            
        # 2. Greedy Expansion
        current_code = list(current_lin)
        existing_set = set(current_code)
        
        if len(current_lin) == 256:
            # No expansion possible
            if len(current_lin) > best_size:
                best_code = current_lin
                best_size = len(current_lin)
            continue
            
        # Compute min_dist to current_lin for all x not in existing_set
        # Use NumPy for speed
        # x_arr: (max_val, 1)
        # lin_arr: (1, len(current_lin))
        # dists: (max_val, len(current_lin))
        
        # This might be memory intensive if len(current_lin) is large, but max is 128 for k=7.
        # 131072 * 128 = 16M bytes. OK.
        
        x_arr = np.arange(max_val)[:, None]
        lin_arr = np.array(current_lin)[None, :]
        
        # Compute XOR
        xor_vals = x_arr ^ lin_arr
        # Flatten to use weights lookup?
        # weights is 1D.
        # dists = weights[xor_vals]
        # This creates a 2D array of distances.
        
        dists = weights[xor_vals]
        
        # min_dist per row
        min_dists = np.min(dists, axis=1)
        
        # Filter candidates
        # We want x such that x not in existing_set and min_dists[x] >= MIN_D
        # Create mask
        mask = np.ones(max_val, dtype=bool)
        for c in existing_set:
            mask[c] = False
            
        valid_indices = np.where(mask & (min_dists >= MIN_D))[0]
        
        # Get corresponding min_dists
        candidate_scores = min_dists[valid_indices]
        
        # Sort by score descending
        # np.argsort
        order = np.argsort(candidate_scores)[::-1]
        
        sorted_candidates = valid_indices[order]
        
        # Greedy Add
        for x in sorted_candidates:
            if time.time() > deadline - 2:
                break
                
            # Check against current_code
            # We can use the precomputed dists? No, current_code changes.
            # We must check against the growing list.
            
            valid = True
            # Optimization: check against newly added words first?
            # Or just all.
            
            # Convert current_code to array for vectorized check?
            # current_code is a list.
            # If len(current_code) is small, loop is fine.
            # If large, vectorize.
            
            if len(current_code) > 50:
                cc_arr = np.array(current_code)
                d = weights[x ^ cc_arr]
                if np.min(d) < MIN_D:
                    continue
            else:
                for c in current_code:
                    if weights[x ^ c] < MIN_D:
                        valid = False
                        break
                if not valid:
                    continue
                    
            current_code.append(x)
            existing_set.add(x)
            
        if len(current_code) > best_size:
            best_code = current_code
            best_size = len(current_code)
            
    return best_code
