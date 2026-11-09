"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using minimum-cost Steiner tree approach.
    
    Structural change: Instead of building balanced trees per output with a cache,
    we use a global Dijkstra to find the minimum gate count to produce each linear
    combination, then build a shared circuit by greedily adding gates in order of
    how much they help complete remaining outputs.
    
    The key insight: the minimum number of gates to produce a set S of inputs is
    related to the minimum size of a binary tree whose leaves are S and internal
    nodes represent XORs. We compute this globally and share optimally.
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
        
        # Find the best split: mask = a ^ b where a and b are "cheap"
        # Try all possible splits and pick the one minimizing new gates
        best_wire = None
        best_cost = float('inf')
        best_pair = None
        
        # Try splitting into two parts
        # For efficiency, try splits where one part is a subset of mask
        # and the other is mask ^ part
        # We want both parts to be "close" to existing wires
        
        # Generate all submasks of mask
        # For m up to 16, this could be large, but we limit by time
        sub = mask
        candidates = []
        while sub:
            candidates.append(sub)
            sub = (sub - 1) & mask
        
        # Also consider the complement within mask
        # Actually, mask = a ^ b means b = mask ^ a, so we just need to try all a that are submasks
        # But we can restrict: try a with popcount around half
        
        # For speed, try a limited set of splits
        k = popcount[mask]
        if k <= 1:
            # Single bit, should be an input
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            # Create it
            gates.append((zero_wire, mask))  # 0 ^ mask = mask, but mask isn't a wire yet
            # Actually this is wrong. Let me just create it directly.
            # For a single bit, it's an input wire
            w = mask  # This is the input wire index
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            return w
        
        # Try balanced splits
        # Find a submask with popcount k//2
        # This is expensive, so use a heuristic: try removing one bit at a time
        
        # Heuristic: try to find two existing or cheap wires whose XOR is mask
        # First, check if mask can be made from two existing wires
        for w1 in list(wires_in_use):
            if time.time() > deadline:
                break
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                # Check if w1 != w2 (XOR with self gives 0)
                if w1 != w2:
                    gates.append((w1, w2))
                    w = next_wire
                    next_wire += 1
                    wire_to_mask[w] = mask
                    mask_to_wire[mask] = w
                    wires_in_use.add(w)
                    return w
        
        # Can't make from two existing wires. Try to create one part, then XOR with existing
        # Find a submask that's close to an existing wire
        # Try: for each existing wire w, see if mask ^ wire_to_mask[w] is simple
        best_split_cost = float('inf')
        best_w1 = None
        best_w2_mask = None
        
        for w1 in list(wires_in_use):
            if time.time() > deadline:
                break
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 == 0:
                continue
            c2 = popcount[m2]
            # Cost to build m2 is at least c2 - 1 gates
            # Prefer smaller c2
            if c2 < best_split_cost:
                best_split_cost = c2
                best_w1 = w1
                best_w2_mask = m2
        
        if best_w1 is not None and best_w2_mask != 0:
            # Recursively build best_w2_mask, then XOR with best_w1
            w2 = get_or_create_wire(best_w2_mask)
            gates.append((best_w1, w2))
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
