import math

def construct():
    N = 511
    # weights[i] corresponds to vertex i+1
    # weight = 1 / (512 - (i+1)) = 1 / (511 - i)
    weights = [0.0] * N
    for i in range(N):
        weights[i] = 1.0 / (511 - i)
        
    # compat[i] is bitset of indices j such that gcd(i+1, j+1) == 1
    compat = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            vj = j + 1
            if math.gcd(vi, vj) == 1:
                mask |= (1 << j)
        compat[i] = mask
        
    # Precompute chunk sums for fast bitset weight summation
    CHUNK_SIZE = 16
    NUM_CHUNKS = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
    MASK_16 = (1 << CHUNK_SIZE) - 1
    
    # chunk_table[c][val] = sum of weights of bits set in 'val' for chunk c
    chunk_table = [[0.0] * (1 << CHUNK_SIZE) for _ in range(NUM_CHUNKS)]
    
    for c in range(NUM_CHUNKS):
        start_idx = c * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, N)
        for val in range(1 << CHUNK_SIZE):
            s = 0.0
            v = val
            bit_pos = 0
            while v:
                if v & 1:
                    global_idx = start_idx + bit_pos
                    if global_idx < N:
                        s += weights[global_idx]
                v >>= 1
                bit_pos += 1
            chunk_table[c][val] = s
            
    def get_sum(mask):
        s = 0.0
        for c in range(NUM_CHUNKS):
            val = (mask >> (c * CHUNK_SIZE)) & MASK_16
            s += chunk_table[c][val]
        return s
        
    best_score = 0.0
    best_set = []
    
    # Initial mask: all bits 0..N-1 set
    initial_mask = (1 << N) - 1
    initial_sum = sum(weights)
    
    # Recursive BB
    # idx: current index to consider (start from N-1 down to 0)
    # score: current accumulated score
    # mask: available vertices (bits 0..N-1)
    # avail_sum: sum of weights of vertices in mask
    # current_set: list of chosen values
    
    def bb(idx, score, mask, avail_sum, current_set):
        nonlocal best_score, best_set
        
        if idx < 0:
            if score > best_score:
                best_score = score
                best_set = current_set[:]
            return
            
        # Bound check
        if score + avail_sum <= best_score:
            return
            
        v_idx = idx
        v_val = v_idx + 1
        w = weights[v_idx]
        bit = 1 << v_idx
        
        # Check if v_idx is available
        if mask & bit:
            # Option 1: Include v_idx
            new_mask = mask & compat[v_idx]
            removed = mask & ~compat[v_idx]
            rem_sum = get_sum(removed)
            new_sum = avail_sum - rem_sum
            bb(idx - 1, score + w, new_mask, new_sum, current_set + [v_val])
            
            # Option 2: Exclude v_idx
            new_mask_ex = mask & ~bit
            new_sum_ex = avail_sum - w
            bb(idx - 1, score, new_mask_ex, new_sum_ex, current_set)
        else:
            # v_idx is not available, must exclude
            bb(idx - 1, score, mask, avail_sum, current_set)
            
    bb(N - 1, 0.0, initial_mask, initial_sum, [])
    
    return best_set
