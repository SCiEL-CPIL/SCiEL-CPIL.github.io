# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit using Dynamic Programming to find the minimum
    gate count for each possible bitmask, and then constructing a shared
    sub-circuit based on these optimal decompositions.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    mat = np.array(matrix, dtype=np.uint8)
    n_outputs, n_inputs = mat.shape

    if n_outputs == 0 or n_inputs == 0:
        return {'gates': [], 'outputs': [n_inputs] * n_outputs}

    # Convert rows to bitmasks
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m

    target_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    # If n_inputs is large, 2^n might be too big for DP.
    # But task says 8x8, 12x16, 16x16. So n_inputs <= 16.
    # 2^16 = 65536. 3^16 ~ 43M ops. This is fine.
    
    if n_inputs > 16:
        # Fallback to simple greedy if n is too large for DP
        # But for this task, n is small.
        pass

    size = 1 << n_inputs
    cost = [0] * size
    split_a = [0] * size
    split_b = [0] * size

    # Initialize costs for inputs and zero
    # cost[0] = 0
    # cost[1 << i] = 0
    
    # DP to compute min cost and optimal split for each mask
    # We iterate masks in increasing order? No, cost depends on submasks.
    # Submasks are smaller in value? Not necessarily (e.g. 1010, submask 1000 is smaller, 0010 is smaller).
    # Actually, any proper submask of m has a value < m?
    # No. 1010 (10). Submask 1000 (8) < 10. 0010 (2) < 10.
    # Yes, any proper subset of bits has a smaller integer value?
    # Let m = sum b_i 2^i. Submask s = sum c_i 2^i where c_i <= b_i.
    # If s != m, then there is some k where b_k=1, c_k=0.
    # And for all j > k, c_j = b_j.
    # So s = m - 2^k + (lower bits diff).
    # Since 2^k is the highest bit where they differ, and m has 1, s has 0, m > s.
    # So yes, proper submasks have smaller integer values.
    # So we can iterate m from 1 to size-1.

    for m in range(1, size):
        # Check if m is a power of 2
        if m & (m - 1) == 0:
            cost[m] = 0
            continue
            
        min_cost = float('inf')
        best_a = 0
        best_b = 0
        
        # Iterate over all non-empty proper submasks
        # To avoid double counting (a, b) and (b, a), we can enforce a < b
        # But since we just need one optimal split, and cost is symmetric,
        # we can just iterate all submasks.
        # Optimization: iterate submasks s of m such that s < m ^ s ?
        # Let's just iterate all to be safe, 3^n is small.
        
        sub = (m - 1) & m
        while sub:
            # sub is a proper non-empty submask
            b = m ^ sub
            # To avoid duplicate work, we can skip if sub > b?
            # If we do, we might miss the case where sub > b but cost is same?
            # No, cost(sub) + cost(b) is symmetric.
            # So we can enforce sub < b to halve iterations.
            if sub < b:
                c = cost[sub] + cost[b] + 1
                if c < min_cost:
                    min_cost = c
                    best_a = sub
                    best_b = b
            sub = (sub - 1) & m
            
        # Edge case: if no sub < b found?
        # This happens if m has only 1 bit? No, we handled powers of 2.
        # If m has 2 bits, e.g., 11. sub=1, b=2. 1<2. OK.
        # If m has 3 bits, e.g., 111.
        # sub=1, b=6. 1<6.
        # sub=2, b=5. 2<5.
        # sub=4, b=3. 4>3. Skip.
        # sub=3, b=4. 3<4.
        # sub=5, b=2. 5>2. Skip.
        # sub=6, b=1. 6>1. Skip.
        # So we cover all pairs.
        
        if min_cost == float('inf'):
            # Should not happen for non-power-of-2
            # Fallback: split into first bit and rest
            lsb = m & (-m)
            best_a = lsb
            best_b = m ^ lsb
            min_cost = cost[best_a] + cost[best_b] + 1
            
        cost[m] = min_cost
        split_a[m] = best_a
        split_b[m] = best_b

    # Check time
    if time.time() - start_time > DEADLINE:
        # Return a trivial solution or best effort
        # Just return inputs if possible? No, must be valid.
        # We will try to finish synthesis quickly.
        pass

    # Synthesize the circuit
    wires = {}
    gates = []
    next_wire = n_inputs + 1
    
    # Pre-fill input wires
    wires[0] = n_inputs
    for i in range(n_inputs):
        wires[1 << i] = i

    def get_wire(mask):
        nonlocal next_wire
        if mask in wires:
            return wires[mask]
        
        # If mask is 0 or input, it's already in wires (handled above)
        # So this is for computed masks
        
        a = split_a[mask]
        b = split_b[mask]
        
        w1 = get_wire(a)
        w2 = get_wire(b)
        
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        wires[mask] = w
        return w

    # Generate wires for all targets
    for mask in target_masks:
        get_wire(mask)

    outputs = [wires[mask] for mask in target_masks]

    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
