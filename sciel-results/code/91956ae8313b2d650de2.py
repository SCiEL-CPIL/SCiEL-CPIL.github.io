def synthesize(matrix, depth_limit):
    import numpy as np
    
    m, n = matrix.shape
    mat = matrix.astype(np.uint8)
    
    gates = []
    next_wire = n + 1
    cache = {}
    wire_to_subset = {}
    
    # Initialize cache with inputs and zero
    cache[frozenset()] = n
    wire_to_subset[n] = frozenset()
    for j in range(n):
        cache[frozenset([j])] = j
        wire_to_subset[j] = frozenset([j])
        
    def xor_gates(w1, w2):
        nonlocal next_wire
        if w1 == w2:
            return n
        if w1 == n:
            return w2
        if w2 == n:
            return w1
            
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        return w

    def estimate_cost(subset):
        if subset in cache:
            return 0
        if len(subset) == 0:
            return 0
        if len(subset) == 1:
            return 0
        return len(subset) // 2

    def build(subset):
        if subset in cache:
            return cache[subset]
            
        if len(subset) == 0:
            return n
        if len(subset) == 1:
            return list(subset)[0]
            
        best_w = None
        best_diff = None
        best_cost = float('inf')
        
        # Evaluate all existing wires
        for w, sub_w in wire_to_subset.items():
            if w == n:
                continue
            diff = subset ^ sub_w
            c = estimate_cost(diff)
            if c < best_cost:
                best_cost = c
                best_w = w
                best_diff = diff
                
        # Evaluate half-split
        lst = sorted(subset)
        mid = len(lst) // 2
        left = frozenset(lst[:mid])
        right = frozenset(lst[mid:])
        split_cost = estimate_cost(left) + estimate_cost(right)
        
        # Decide
        if best_w is not None and (best_cost + 1) <= (split_cost + 1):
            # Use best_w
            if best_diff in cache:
                w_other = cache[best_diff]
            elif len(best_diff) == 1:
                w_other = list(best_diff)[0]
            elif len(best_diff) == 0:
                w_other = n
            else:
                w_other = build(best_diff)
                
            w_new = xor_gates(best_w, w_other)
            cache[subset] = w_new
            wire_to_subset[w_new] = subset
            return w_new
        else:
            # Use half-split
            w_left = build(left)
            w_right = build(right)
            w_new = xor_gates(w_left, w_right)
            cache[subset] = w_new
            wire_to_subset[w_new] = subset
            return w_new

    # Sort rows by popcount to build simple combinations first
    row_indices = list(range(m))
    row_indices.sort(key=lambda i: np.sum(mat[i]))
    
    outputs = [0] * m
    for i in row_indices:
        indices = np.where(mat[i] == 1)[0]
        subset = frozenset(indices)
        w = build(subset)
        outputs[i] = w
        
    return {'gates': gates, 'outputs': outputs}
