def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N  # 131072

    # Precompute weights for all 17-bit words
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    def get_linear_code():
        """
        Construct the [17,8,6] linear code (256 words).
        This is the shortening of the [18,9,6] quadratic residue code.
        
        The generator matrix for the [17,8,6] code can be derived from the QR code.
        For the prime 17, the quadratic residues are {1, 2, 4, 8, 9, 13, 15, 16}.
        
        A standard generator matrix G (8x17) for a [17,8,6] code with minimum distance 6
        can be constructed. Let's use a known systematic form or a specific basis.
        
        Actually, let's just use a greedy search to find 8 basis vectors that form
        a valid [17,8,6] code. Since the space is small and we know such a code exists,
        a simple greedy approach will work fast enough.
        """
        basis = []
        code = [0]
        
        # We'll iterate through candidates in a specific order to ensure we find a full-rank basis
        # that maintains distance >= 6.
        # To speed up, we can try to find high-weight vectors first, or just iterate.
        # Iterating 1 to SIZE-1 and checking against current code (size 2^k) is fast.
        
        for w in range(1, SIZE):
            if len(basis) == 8:
                break
            if time.time() > deadline - 50:
                break
            
            # Check if w is at distance >= 6 from all c in current_code
            valid = True
            for c in code:
                if weights[w ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                basis.append(w)
                # Extend code
                new_words = [c ^ w for c in code]
                code.extend(new_words)
        
        return code

    # Step 1: Get the 256-word linear code
    lin_code = get_linear_code()
    
    if len(lin_code) < 256:
        # Fallback: return what we have
        return lin_code

    # Step 2: Find candidates outside the linear code
    # A word w is a candidate if min_{c in lin_code} d(w, c) >= 6
    
    lin_code_np = np.array(lin_code, dtype=np.int32)
    lin_code_set = set(lin_code)
    
    # Vectorized distance check to all linear code words
    # For each w, compute distances to all 256 lin_code words
    # We can do this in batches or one by one. One by one with numpy is fast.
    
    candidates = []
    # We can vectorize the check for all w at once?
    # weights[w ^ lin_code_np] gives an array of 256 distances.
    # We need min >= 6.
    
    # Let's iterate w from 0 to SIZE-1, skipping lin_code words.
    # 131072 iterations, each doing a numpy op of size 256.
    # This should be fast (~131k * small overhead).
    
    for w in range(SIZE):
        if w in lin_code_set:
            continue
        
        dists = weights[w ^ lin_code_np]
        if dists.min() >= MIN_D:
            candidates.append(w)
            
    if not candidates:
        return lin_code

    # Step 3: Find maximum subset of candidates with pairwise distance >= 6
    # This is a Max Independent Set problem.
    # We want to find as many candidates as possible that are mutually at distance >= 6.
    
    # Strategy: Greedy with random restarts.
    # Since we expect only a few (maybe 2-10) candidates can be added,
    # the search space is small in terms of the final set size.
    
    best_subset = []
    max_found = 0
    
    num_restarts = 500  # Can be increased if time permits
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        cands = candidates[:]
        random.shuffle(cands)
        
        current_set = []
        for w in cands:
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
            
            # If we found a decent number, we can potentially stop early
            # but let's keep going to see if we can find more.
            # If max_found is large, the greedy is likely good.

    final_code = lin_code + best_subset
    return final_code
