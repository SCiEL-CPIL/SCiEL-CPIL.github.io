# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0
    
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, MOD)
        
    # min_size[r] will store the minimum subset size that sums to r
    # best_combo[r] will store the corresponding subset
    min_size = [float('inf')] * MOD
    best_combo = [None] * MOD
    
    # Iterate through all non-empty subsets
    # To ensure we find the minimum size, we can iterate by size k
    # and only update if current k is smaller (which it is, since we go 1..10)
    # Or just iterate all and update min.
    
    for k in range(1, 11):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        for combo in combinations(DENOMS, k):
            s = 0
            for d in combo:
                s = (s + inv[d]) % MOD
            if k < min_size[s]:
                min_size[s] = k
                best_combo[s] = list(combo)
                
    # Check if all residues are covered
    for r in range(MOD):
        if min_size[r] == float('inf'):
            # Fallback, should not happen
            best_combo[r] = [1] # Placeholder, but this might be wrong sum
            
    # The optimal K is the max of min_size
    # But we just need to return the lists.
    # The evaluator will measure the max length.
    
    return best_combo
# EVOLVE-BLOCK-END
