"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using a shared subexpression cache with balanced tree construction.
    
    Key improvements over the previous approach:
    1. Use a global cache of (mask -> wire) so that shared subexpressions are built only once.
    2. For each new mask, find the best split (a, b) where a^b = mask, minimizing the
       total new gates needed. Prefer splits where both a and b are already available or
       have low construction cost.
    3. Process outputs in order of increasing popcount to build simpler subexpressions first.
    4. Use a balanced split strategy: for a mask with k bits, split into two submasks of
       sizes floor(k/2) and ceil(k/2) to minimize tree height and enable sharing.
    """
    deadline = time.time() + 80  # 10s safety margin
    
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    
    if n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to bitmasks
    rows = []
    for i in range(n):
        mask = 0
        for j in range(m):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Zero wire is m, inputs are 0..m-1
    zero_wire = m
    next_wire = m + 1
    gates = []
    
    # Wire management
    wire_to_mask = {}  # wire_id -> bitmask
    mask_to_wire = {}  # bitmask -> wire_id
    for j in range(m):
        wire_to_mask[j] = 1 << j
        mask_to_wire[1 << j] = j
    wire_to_mask[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    # Precompute popcount for all possible masks (m <= 16)
    popcount = [0] * (1 << m)
    for i in range(1, 1 << m):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Sort outputs by complexity (popcount)
    output_indices = list(range(n))
    output_indices.sort(key=lambda i: popcount[rows[i]])
    
    outputs = [0] * n
    wires_in_use = set(wire_to_mask.keys())
    
    def get_or_create_wire(mask):
        """Get wire for mask, or create one using minimum new gates."""
        nonlocal next_wire
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return zero_wire
        
        k = popcount[mask]
        if k == 1:
            # Single bit, should be an input wire
            w = mask  # The input wire index for this single bit
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
            return w
        
        # Try to find a split mask = a ^ b where both a and b are available or cheap
        # Strategy: try all submasks of mask, find the one that minimizes cost
        # Cost of a submask s is 0 if it's already available, else estimated as popcount[s] - 1
        
        # First, check if mask can be formed by XOR of two existing wires
        for w1 in list(wires_in_use):
            if time.time() > deadline:
                break
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 != w2:
                    gates.append((w1, w2))
                    w = next_wire
                    next_wire += 1
                    wire_to_mask[w] = mask
                    mask_to_wire[mask] = w
                    wires_in_use.add(w)
                    return w
        
        # Find the best split: iterate over submasks of mask
        # For each submask a, b = mask ^ a
        # We want to minimize the "new cost" = cost_to_build(a) + cost_to_build(b)
        # where cost_to_build(x) = 0 if x is available, else popcount[x] - 1 (lower bound)
        
        best_a = None
        best_cost = float('inf')
        
        # Iterate over all submasks of mask
        sub = mask
        while sub:
            if time.time() > deadline:
                break
            a = sub
            b = mask ^ sub
            if a == b:
                sub = (sub - 1) & mask
                continue
            
            # Estimate cost
            cost_a = 0 if a in mask_to_wire else (popcount[a] - 1)
            cost_b = 0 if b in mask_to_wire else (popcount[b] - 1)
            total_cost = cost_a + cost_b
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_a = a
            
            sub = (sub - 1) & mask
        
        if best_a is not None:
            a = best_a
            b = mask ^ best_a
            # Recursively build a and b, then XOR them
            w_a = get_or_create_wire(a)
            w_b = get_or_create_wire(b)
            gates.append((w_a, w_b))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
            return w
        
        # Fallback: build from scratch using a chain
        bits = [j for j in range(m) if mask & (1 << j)]
        if not bits:
            return zero_wire
        w = bits[0]
        for j in bits[1:]:
            gates.append((w, j))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
        return w
    
    # Process outputs in order of increasing complexity
    for i in output_indices:
        if time.time() > deadline:
            # Emergency fallback
            for remaining in output_indices[output_indices.index(i):]:
                if outputs[remaining] != 0:
                    continue
                bits = [j for j in range(m) if rows[remaining] & (1 << j)]
                if not bits:
                    outputs[remaining] = zero_wire
                elif len(bits) == 1:
                    outputs[remaining] = bits[0]
                else:
                    w = bits[0]
                    for j in bits[1:]:
                        gates.append((w, j))
                        w = next_wire
                        next_wire += 1
                    outputs[remaining] = w
            break
        
        w = get_or_create_wire(rows[i])
        outputs[i] = w
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
