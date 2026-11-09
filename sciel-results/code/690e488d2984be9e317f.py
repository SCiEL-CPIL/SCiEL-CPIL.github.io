import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    N = 17
    MIN_D = 6
    
    # Precompute weights for all 2^N words
    # Using list for fast access
    max_val = 1 << N
    weights = [0] * max_val
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    # Phase 1: Construct a Linear Code
    # We want to find a basis such that the code size is 2^k
    # Greedy approach: add vectors that maintain min distance
    basis = []
    current_code = [0]
    
    # We aim for 8 basis vectors (size 256)
    # To speed up, we can scan candidates
    # A good order for linear code basis is often vectors with high weight or specific structures,
    # but simple scan works if we break early.
    
    # Optimization: 
    # When checking candidate g against current_code, if we find any c such that 
    # weight(g ^ c) < MIN_D, we reject.
    
    # We can try to find 8 rows.
    # We scan 1 to max_val-1.
    
    for g in range(1, max_val):
        if time.time() > deadline - 30: # Leave 30s for phase 2
            break
            
        valid = True
        # Check against all existing codewords
        for c in current_code:
            if weights[g ^ c] < MIN_D:
                valid = False
                break
                
        if valid:
            basis.append(g)
            # Expand code
            # New code is {c, c^g for c in current_code}
            old_code = current_code
            current_code = []
            for c in old_code:
                current_code.append(c)
                current_code.append(c ^ g)
                
            if len(basis) >= 8:
                break
                
    # Phase 2: Non-linear Expansion with improved greedy
    existing_set = set(current_code)
    code_list = list(current_code)  # working copy
    
    # Build a distance lookup structure for faster queries
    # For each codeword c, precompute which x are at distance < MIN_D
    # But that's expensive. Instead, we'll use a smarter greedy.
    
    # Compute min distance from each x to the current code
    # Only consider x with min_dist >= MIN_D
    candidates = []
    
    for x in range(max_val):
        if x in existing_set:
            continue
        
        min_d = max_val
        for c in code_list:
            d = weights[x ^ c]
            if d < min_d:
                min_d = d
                if min_d < MIN_D:
                    break
        
        if min_d >= MIN_D:
            candidates.append((min_d, x))
    
    # Sort by min distance descending - prefer words far from existing code
    candidates.sort(key=lambda item: -item[0])
    
    # Greedy addition with full validation
    added_count = 0
    for min_d, x in candidates:
        if time.time() > deadline - 3:
            break
        
        # Check against ALL current codewords (not just added_in_phase2)
        valid = True
        for c in code_list:
            if weights[x ^ c] < MIN_D:
                valid = False
                break
        
        if valid:
            code_list.append(x)
            existing_set.add(x)
            added_count += 1
    
    result = code_list
    return result
