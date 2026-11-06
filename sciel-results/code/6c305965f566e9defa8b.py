import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Values 2 to 256
    N = 256
    values = list(range(2, N + 1))
    num_nodes = len(values)
    
    # Precompute conflict bitmasks
    # conflict[i] has bit j set if lcm(values[i], values[j]) <= 256
    conflict = [0] * num_nodes
    for i in range(num_nodes):
        a = values[i]
        for j in range(i + 1, num_nodes):
            b = values[j]
            if math.lcm(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    # Weights: 1/v
    weights = [1.0 / v for v in values]
    
    def get_score(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # Heuristic 1: Start with 2 and all large odds
    # Index of 2 is 0 (since values[0]=2)
    # Large odds: values 129..256 where v % 2 != 0
    mask1 = 0
    mask1 |= (1 << 0) # Add 2
    for i, v in enumerate(values):
        if v >= 129 and v % 2 != 0:
            mask1 |= (1 << i)
            
    # Heuristic 2: Start with all large numbers
    mask2 = 0
    for i, v in enumerate(values):
        if v >= 129:
            mask2 |= (1 << i)
            
    # Heuristic 3: Greedy by weight (1/v desc, i.e., v asc)
    mask3 = 0
    order = sorted(range(num_nodes), key=lambda i: -weights[i])
    for i in order:
        if (conflict[i] & mask3) == 0:
            mask3 |= (1 << i)
            
    best_mask = max([mask1, mask2, mask3], key=get_score)
    best_score = get_score(best_mask)
    
    # Local Search: 1-for-1 swaps
    def can_add(mask, i):
        return (conflict[i] & mask) == 0
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one node
        for i in range(num_nodes):
            if (best_mask >> i) & 1:
                new_mask = best_mask & ~(1 << i)
                # Try adding any other node
                for j in range(num_nodes):
                    if j == i or ((best_mask >> j) & 1):
                        continue
                    if can_add(new_mask, j):
                        candidate = new_mask | (1 << j)
                        score = get_score(candidate)
                        if score > best_score + 1e-12:
                            best_mask = candidate
                            best_score = score
                            improved = True
                            break
                if improved:
                    break
        
        # Try adding one node (if possible)
        if not improved:
            for i in range(num_nodes):
                if not ((best_mask >> i) & 1) and can_add(best_mask, i):
                    candidate = best_mask | (1 << i)
                    score = get_score(candidate)
                    if score > best_score + 1e-12:
                        best_mask = candidate
                        best_score = score
                        improved = True
                        break
                        
    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(values[idx])
        m ^= lsb
        
    return result
