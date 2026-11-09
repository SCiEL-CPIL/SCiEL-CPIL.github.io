import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights for all 2^17 words
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)
    
    # Precompute offsets for distances < MIN_D
    # offsets[d] contains all integers x such that x.bit_count() == d
    offsets = [[] for _ in range(MIN_D)]
    for i in range(SIZE):
        w = i.bit_count()
        if w < MIN_D:
            offsets[w].append(i)
    
    # Combine into a single list of all forbidden offsets
    forbidden_offsets = []
    for d in range(MIN_D):
        forbidden_offsets.extend(offsets[d])
    forbidden_offsets = np.array(forbidden_offsets, dtype=np.int32)
    
    def get_linear_code():
        """
        Construct the [17,8,6] linear code (256 words).
        """
        code = [0]
        code_np = np.array(code, dtype=np.int32)
        
        for w in range(1, SIZE):
            if time.time() > deadline - 30:
                break
            if len(code) == 256:
                break
            
            # Check if w is at distance >= MIN_D from all current codewords
            # Vectorized check
            dists = weights[w ^ code_np]
            if dists.min() >= MIN_D:
                # Add w to the code
                code.append(w)
                # Extend the numpy array for vectorized checks
                new_words = code_np ^ w
                code_np = np.concatenate([code_np, new_words])
                
        return code

    # Phase 1: Get the linear code
    lin_code = get_linear_code()
    
    if len(lin_code) < 256:
        return lin_code

    lin_code_set = set(lin_code)
    
    # Phase 2: Find nonlinear extensions using forbidden zone marking
    
    # Initialize blocked array
    # A word is blocked if it is in the linear code OR is at distance < 6 from a linear code word
    blocked = np.zeros(SIZE, dtype=bool)
    
    # Mark linear code words and their neighbors
    # We can do this by iterating through the linear code and marking neighbors
    # But we can be smarter: 
    # For each c in lin_code, mark c ^ offset for all offset in forbidden_offsets
    
    # Vectorized marking is tricky because of overlaps, but boolean OR is idempotent
    # Let's just loop through the 256 linear code words. 
    # 256 * 9400 = 2.4M operations. This is fast.
    
    for c in lin_code:
        # Generate neighbors
        neighbors = c ^ forbidden_offsets
        blocked[neighbors] = True
    
    # Get candidates: words not blocked
    candidates_mask = ~blocked
    candidates = np.where(candidates_mask)[0]
    
    if len(candidates) == 0:
        return lin_code
    
    best_extra = []
    max_extra = 0
    
    # We want to find the largest subset of candidates with pairwise distance >= 6.
    # We use a greedy approach with random restarts.
    # Since the number of candidates is likely small (a few hundred or thousand),
    # and the number of extra words is small (2-4), this is efficient.
    
    num_restarts = 2000 # Can be high because each restart is fast
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Shuffle candidates
        cands = candidates.copy()
        random.shuffle(cands)
        
        current_extra = []
        # We maintain a local blocked array for this restart? 
        # No, we can just check against current_extra.
        # But current_extra is small.
        # However, we can reuse the "blocked" logic locally for speed if needed,
        # but simple loop is fine for small sets.
        
        for w in cands:
            # Check if w is compatible with all in current_extra
            valid = True
            for u in current_extra:
                if weights[w ^ u] < MIN_D:
                    valid = False
                    break
            if valid:
                current_extra.append(w)
                
        if len(current_extra) > max_extra:
            max_extra = len(current_extra)
            best_extra = current_extra[:]
            
            # If we found a good number, we might stop early?
            # Theoretical max is 258, so 2 extra. If we find 2, we can stop?
            # No, maybe there are more? Upper bound is 340, but 258 is the known best.
            # Let's keep searching for a bit to be sure, but 2 is likely the max.
            
    final_code = lin_code + best_extra
    return final_code
