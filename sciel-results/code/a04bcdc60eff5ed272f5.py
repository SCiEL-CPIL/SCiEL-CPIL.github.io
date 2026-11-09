"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using greedy sub-expression sharing with
    minimum-cost split selection.
    
    Strategy:
    1. Collect all unique output masks.
    2. Sort by complexity (popcount) to build simpler expressions first.
    3. For each new mask needed, find the split (a, b) such that a^b = mask
       minimizing the number of NEW gates required.
    4. Prefer splits where both a and b are already available (0 new gates for children).
    5. If not available, recursively build the cheaper child first.
    6. Use a global cache to share sub-expressions across outputs.
    """
    deadline = time.time() + 50  # 10s safety margin for 60s limit
    
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
    
    # Sort outputs by complexity (popcount) - simpler ones first to maximize sharing
    output_indices = list(range(n))
    output_indices.sort(key=lambda i: popcount[rows[i]])
    
    outputs = [0] * n
    wires_in_use = set(wire_to_mask.keys())
    
    # Track gate count for cost estimation
    gate_count = 0
    
    def get_or_create_wire(mask):
        """Get wire for mask, or create one using minimum new gates."""
        nonlocal next_wire, gate_count
        
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return zero_wire
        
        k = popcount[mask]
        if k <= 1:
            # Single bit, should be an input
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            # For a single bit, it's an input wire
            w = mask  # This is the input wire index
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            return w
        
        # Try to find the best split: mask = a ^ b
        # We want to minimize the total new gates needed.
        # Strategy: try all submasks of mask, but prioritize balanced splits
        # and those where the components are already available.
        
        best_cost = float('inf')
        best_a = None
        best_b = None
        
        # Generate submasks of mask
        # For efficiency, we iterate through submasks but limit the search
        # by checking time and focusing on "interesting" splits
        
        # First pass: check if mask can be formed by XOR of two EXISTING wires
        # This would cost 1 gate total (just the final XOR)
        existing_wires = list(wires_in_use)
        for w1 in existing_wires:
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 != w2:  # Avoid XOR with self
                    # Cost is 1 gate
                    if 1 < best_cost:
                        best_cost = 1
                        best_a = w1
                        best_b = w2
        
        if best_cost == 1:
            # Found a perfect split using existing wires
            gates.append((best_a, best_b))
            gate_count += 1
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
            return w
        
        # Second pass: find the split that minimizes recursive cost
        # We estimate cost as the number of new gates needed for a and b
        # A simple heuristic: cost(mask) ≈ popcount(mask) - 1
        # But we can do better by checking if submasks are already available
        
        # Iterate through submasks of mask
        # To keep it fast, we'll try a limited set of submasks
        # Focus on submasks with popcount around k/2
        
        sub = mask
        found_split = False
        
        # Limit the number of submasks we check to avoid timeout
        max_checks = 200 if m > 12 else 1000
        
        check_count = 0
        while sub:
            check_count += 1
            if check_count > max_checks:
                break
            
            if time.time() > deadline:
                break
            
            comp = mask ^ sub
            if comp == 0:
                sub = (sub - 1) & mask
                continue
            
            # Cost to build sub: 0 if available, else estimate
            if sub in mask_to_wire:
                cost_sub = 0
            else:
                cost_sub = popcount[sub] - 1  # Estimate
            
            # Cost to build comp: 0 if available, else estimate
            if comp in mask_to_wire:
                cost_comp = 0
            else:
                cost_comp = popcount[comp] - 1  # Estimate
            
            # Total cost for this split: cost_sub + cost_comp + 1 (for the XOR gate)
            total_cost = cost_sub + cost_comp + 1
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_a = sub
                best_b = comp
            
            sub = (sub - 1) & mask
        
        if best_a is not None:
            # Recursively build the two components
            w1 = get_or_create_wire(best_a)
            w2 = get_or_create_wire(best_b)
            
            # Create the final XOR gate
            gates.append((w1, w2))
            gate_count += 1
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
            gate_count += 1
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
        return w
    
    # Process outputs in order of increasing complexity
    for i in output_indices:
        if time.time() > deadline:
            # Emergency fallback for remaining outputs
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
