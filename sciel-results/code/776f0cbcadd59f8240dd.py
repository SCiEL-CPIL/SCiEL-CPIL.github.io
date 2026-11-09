def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 88  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    
    # Precompute weights for all 2^17 words using numpy
    all_words = np.arange(1 << N, dtype=np.int64)
    
    # Use a precomputed table for 8-bit chunks
    pop_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
    
    # Vectorized weight calculation
    w = all_words
    weights = (pop_table[w & 0xFF] + 
               pop_table[(w >> 8) & 0xFF] + 
               pop_table[(w >> 16) & 0xFF])
    
    # Precompute all masks with weight < MIN_D
    masks = []
    for m in range(1 << N):
        if weights[m] < MIN_D:
            masks.append(m)
    masks = np.array(masks, dtype=np.int64)
    
    # Helper function to build a linear code greedily
    def build_linear_code(seed_order, max_basis=8):
        current_code = np.array([0], dtype=np.int64)
        basis = []
        
        for g in seed_order:
            if time.time() > deadline:
                break
            
            # Check if g can be added
            xor_vals = g ^ current_code
            min_dist = np.min(weights[xor_vals])
            
            if min_dist >= MIN_D:
                basis.append(g)
                new_code = np.concatenate([current_code, current_code ^ g])
                current_code = new_code
                
                if len(basis) >= max_basis:
                    break
        
        return current_code, basis
    
    # Helper function to extend code with nonlinear words
    def extend_code(linear_code):
        # 1. Find all words at distance >= MIN_D from all codewords in linear_code
        min_dists = np.full(1 << N, 100, dtype=np.int8)
        for c in linear_code:
            # Vectorized distance calculation
            d = weights[all_words ^ c]
            min_dists = np.minimum(min_dists, d)
            
        candidates = np.where(min_dists >= MIN_D)[0]
        
        if len(candidates) == 0:
            return set(linear_code.tolist())
        
        # 2. Greedily select candidates that are mutually compatible
        # We want to maximize the number of added words.
        # Since we expect few candidates, a simple greedy is sufficient.
        added = []
        for x in candidates:
            # Check compatibility with already added candidates
            compatible = True
            for y in added:
                if weights[x ^ y] < MIN_D:
                    compatible = False
                    break
            if compatible:
                added.append(x)
                
        result = set(linear_code.tolist())
        for x in added:
            result.add(x)
            
        return result

    # Try multiple strategies to find the best code
    
    best_result = None
    
    # Strategy 1: Standard linear code + greedy extension
    # Try different orderings for the linear code construction
    
    # First, try the standard order (1, 2, 3, ...)
    standard_order = list(range(1, 1 << N))
    linear_code, basis = build_linear_code(standard_order)
    
    if len(basis) >= 8:
        # We have a [17,8,6] code with 256 words
        result = extend_code(linear_code)
        
        if best_result is None or len(result) > len(best_result):
            best_result = result
    
    # Strategy 2: Try randomized linear code constructions
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Randomize the order of candidates
        random_order = np.random.permutation(range(1, 1 << N))
        linear_code, basis = build_linear_code(random_order)
        
        if len(basis) >= 8:
            result = extend_code(linear_code)
            
            if best_result is None or len(result) > len(best_result):
                best_result = result
    
    # If we didn't find a good code, fall back to simple greedy
    if best_result is None or len(best_result) < 256:
        # Simple greedy: just add words in order if they fit
        current_code = np.array([0], dtype=np.int64)
        added_set = {0}
        forbidden = np.zeros(1 << N, dtype=bool)
        forbidden[masks] = True
        
        for w in range(1, 1 << N):
            if time.time() > deadline:
                break
            
            if not forbidden[w]:
                added_set.add(w)
                forbidden[w ^ masks] = True
        
        best_result = added_set
    
    return list(best_result)
