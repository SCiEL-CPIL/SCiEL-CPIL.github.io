def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = matrix.shape[1]
    m = matrix.shape[0]
    
    # Convert to masks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j] != 0:
                mask |= (1 << j)
        targets.append(mask)
    
    # Sort by popcount descending for better sharing
    target_indices = sorted(range(m), key=lambda i: targets[i].bit_count(), reverse=True)
    
    gates = []
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n  # zero wire
    
    def get_bits(mask):
        bits = []
        tmp = mask
        while tmp:
            lb = tmp & (-tmp)
            bits.append(lb)
            tmp ^= lb
        return bits
    
    def get_wire(mask):
        if mask in memo:
            return memo[mask]
        
        if time.time() > deadline:
            # Fallback: naive
            bits = get_bits(mask)
            if not bits:
                memo[mask] = n
                return n
            if len(bits) == 1:
                memo[mask] = memo[bits[0]]
                return memo[bits[0]]
            w = memo[bits[0]]
            for b in bits[1:]:
                new_wire = n + 1 + len(gates)
                gates.append([w, memo[b]])
                w = new_wire
            memo[mask] = w
            return w
        
        bits = get_bits(mask)
        if not bits:
            memo[mask] = n
            return n
        if len(bits) == 1:
            memo[mask] = memo[bits[0]]
            return memo[bits[0]]
        
        # Split into two roughly equal halves
        k = len(bits)
        half = k // 2
        
        # Try a few split points and pick the best
        best_split = None
        best_cost = float('inf')
        
        for h in [half, half + 1]:
            if 0 < h < k:
                mask1 = 0
                for b in bits[:h]:
                    mask1 |= b
                mask2 = mask ^ mask1
                if mask1 != 0 and mask2 != 0:
                    # Estimate cost
                    c1 = mask1.bit_count()
                    c2 = mask2.bit_count()
                    cost = c1 + c2
                    if cost < best_cost:
                        best_cost = cost
                        best_split = (mask1, mask2)
        
        if best_split is None:
            # Fallback: split off one bit
            mask1 = bits[0]
            mask2 = mask ^ mask1
            best_split = (mask1, mask2)
        
        mask1, mask2 = best_split
        w1 = get_wire(mask1)
        w2 = get_wire(mask2)
        
        new_wire = n + 1 + len(gates)
        gates.append([w1, w2])
        memo[mask] = new_wire
        return new_wire
    
    # Compute all targets
    output_wires = [0] * m
    for idx in target_indices:
        if time.time() > deadline:
            # Compute remaining naively
            for j in range(idx, m):
                bits = get_bits(targets[j])
                if not bits:
                    output_wires[j] = n
                elif len(bits) == 1:
                    output_wires[j] = memo[bits[0]]
                else:
                    w = memo[bits[0]]
                    for b in bits[1:]:
                        new_wire = n + 1 + len(gates)
                        gates.append([w, memo[b]])
                        w = new_wire
                    output_wires[j] = w
            break
        else:
            output_wires[idx] = get_wire(targets[idx])
    
    return {'gates': gates, 'outputs': output_wires}
