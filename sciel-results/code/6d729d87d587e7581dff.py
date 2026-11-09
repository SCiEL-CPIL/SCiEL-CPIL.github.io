def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Precompute weights for all 2^17 words
    # i.bit_count() is fast in Python 3.10+
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    def get_linear_code():
        """
        Construct the [17,8,6] linear code (256 words).
        This is the shortening of the [18,9,6] quadratic residue code.
        """
        basis = []
        code = [0]
        
        # We want to find a basis of size 8 such that all 2^8 linear combinations
        # have weight >= 6.
        # Greedy approach: iterate through all non-zero vectors, try to add to basis.
        
        # Optimization: Sort candidates by weight descending?
        # High weight vectors are "safer" but might not form a good basis.
        # Let's just iterate 1 to SIZE-1. It's deterministic and usually finds a code quickly.
        # If we fail, we can try random restarts, but the first pass is usually enough.
        
        for w in range(1, SIZE):
            if time.time() > deadline - 40:
                break
            if len(basis) == 8:
                break
            
            # Check if w is at distance >= MIN_D from all current codewords
            valid = True
            # code size is 2^len(basis), max 128
            for c in code:
                if weights[w ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                basis.append(w)
                # Extend code
                # new_words = [c ^ w for c in code]
                # code.extend(new_words)
                # Vectorized extension is faster
                code_arr = np.array(code, dtype=np.int32)
                new_words = code_arr ^ w
                code = list(code_arr) + list(new_words)
                
        return code

    # Phase 1: Get the linear code
    lin_code = get_linear_code()
    
    if len(lin_code) < 256:
        # Fallback: return what we have
        # This shouldn't happen with the standard greedy on QR codes, but just in case
        return lin_code

    # Phase 2: Find nonlinear extensions
    # We have a linear code C of size 256.
    # We want to find words w not in C such that min_{c in C} d(w, c) >= 6.
    
    lin_code_set = set(lin_code)
    lin_code_np = np.array(lin_code, dtype=np.int32)
    
    # Identify all candidates outside the linear code that are at distance >= 6 from the linear code
    candidates = []
    
    # Iterate through all words
    # We can skip words in lin_code
    for w in range(SIZE):
        if w in lin_code_set:
            continue
        
        # Check distance to all lin_code words
        # Vectorized check
        dists = weights[w ^ lin_code_np]
        if dists.min() >= MIN_D:
            candidates.append(w)
            
    if not candidates:
        return lin_code

    # We have a list of candidates.
    # We want to find the largest subset of candidates with pairwise distance >= 6.
    # This is a Maximum Independent Set problem on the graph where edges connect words at distance < 6.
    # Since the expected number of added words is small (Milshtein bound is 258, so 2 words),
    # we can use a greedy approach with random restarts to find a good subset.
    
    best_subset = []
    max_found = 0
    
    # Number of restarts depends on remaining time
    # Each restart shuffles candidates and runs greedy
    # Greedy is O(K^2) where K is number of candidates found so far (small)
    # But we iterate through all candidates.
    
    # Let's estimate time per restart.
    # If there are ~1000 candidates, 1000 * (avg subset size) checks.
    # Subset size is small (2-4). So ~4000 checks per restart.
    # 1000 restarts -> 4M checks. Should be fast.
    
    num_restarts = 500  # Safe number, can increase if fast
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        cands = candidates[:]
        random.shuffle(cands)
        
        current_set = []
        for w in cands:
            # Check if w is compatible with all words in current_set
            valid = True
            for u in current_set:
                if weights[w ^ u] < MIN_D:
                    valid = False
                    break
            if valid:
                current_set.append(w)
                
        if len(current_set) > max_found:
            max_found = len(current_set)
            best_subset = current_set[:]
            
        # Early termination if we find a very large set?
        # Unlikely to find > 10, so just keep going until time limit or restarts done.
        
    final_code = lin_code + best_subset
    return final_code
