"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using Dijkstra-based minimum-cost arborescence on the
    hypergraph of linear combinations.
    
    Key insight: We model the problem as finding minimum-cost paths from the zero
    vector to each target output in a graph where vertices are all possible
    linear combinations (bitmasks) and edges represent single XOR gates.
    
    The cost of producing a mask is the minimum number of gates needed. We use
    Dijkstra's algorithm with a key insight: the cost to produce mask M is
    1 + min over all pairs (A, B) where A^B=M of max(cost[A], cost[B]) for depth,
    or 1 + min over all A of cost[A] + cost[M^A] for gate count.
    
    For gate count minimization with sharing, we process masks in order of
    increasing "cost" (popcount-based lower bound) and build the circuit greedily.
    """
    deadline = time.time() + 75  # 15s safety margin
    
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
    
    # Precompute for each mask, the list of submasks to try for splitting
    # We'll use a smarter approach: for each mask, precompute good split points
    
    # Dijkstra-like approach: compute minimum gates needed for each mask
    # dist[mask] = minimum number of gates to produce this mask
    # parent[mask] = (mask_a, mask_b) such that mask = mask_a ^ mask_b and
    #                dist[mask] = 1 + max(dist[mask_a], dist[mask_b]) for depth
    #                or dist[mask] = 1 + dist[mask_a] + dist[mask_b] for gate count
    # 
    # For gate count with sharing, we want to minimize total gates in the circuit.
    # This is equivalent to finding a minimum-cost set of gates that produces all
    # required outputs.
    
    # Simpler approach: use the fact that for a mask with k bits, the minimum
    # gates to produce it alone is k-1. With sharing, we can do better.
    # 
    # Strategy: Process masks in order of increasing popcount. For each mask,
    # find the best split (a, b) where a^b = mask, minimizing the total new
    # gates needed. A split is "free" if both a and b are already available.
    
    # Initialize: all single-bit masks and zero are available at cost 0
    available = set()
    for j in range(m):
        available.add(1 << j)
    available.add(0)
    
    # For each mask, store the best split
    best_split = {}  # mask -> (mask_a, mask_b)
    
    # Process masks in order of increasing popcount
    all_masks = list(range(1, 1 << m))
    all_masks.sort(key=lambda x: popcount[x])
    
    # For each mask, find the best split
    for mask in all_masks:
        if mask in available:
            continue
        if time.time() > deadline:
            break
        
        k = popcount[mask]
        if k <= 1:
            continue
        
        # Try all possible splits: mask = a ^ b
        # a must be a proper submask of mask (a != 0, a != mask)
        # b = mask ^ a
        best_cost = float('inf')
        best_a = None
        best_b = None
        
        # Iterate over all non-trivial submasks
        # Optimization: only consider submasks with popcount between 1 and k-1
        # and prefer balanced splits
        sub = mask
        while sub:
            if time.time() > deadline:
                break
            if sub == mask:
                sub = (sub - 1) & mask
                continue
            
            b = mask ^ sub
            if b == 0:
                sub = (sub - 1) & mask
                continue
            
            # Cost of this split:
            # If both sub and b are available, cost = 0 (just add one gate)
            # If one is available, cost = cost of producing the other
            # If neither is available, cost = cost of producing both
            
            cost_a = 0 if sub in available else popcount[sub] - 1
            cost_b = 0 if b in available else popcount[b] - 1
            total_cost = cost_a + cost_b + 1  # +1 for the XOR gate itself
            
            # Prefer splits where both parts are simpler
            # Also prefer balanced splits
            balance = abs(popcount[sub] - popcount[b])
            
            # Use a composite score
            score = total_cost * 10 + balance
            
            if score < best_cost:
                best_cost = score
                best_a = sub
                best_b = b
            
            sub = (sub - 1) & mask
        
        if best_a is not None:
            best_split[mask] = (best_a, best_b)
    
    # Now build the circuit
    # Process outputs in order of increasing complexity
    output_indices = list(range(n))
    output_indices.sort(key=lambda i: popcount[rows[i]])
    
    outputs = [0] * n
    wires_in_use = set(wire_to_mask.keys())
    
    def get_or_create_wire(mask):
        """Get wire for mask, or create one using the precomputed best split."""
        nonlocal next_wire
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return zero_wire
        
        # Use the precomputed best split
        if mask in best_split:
            a, b = best_split[mask]
            w_a = get_or_create_wire(a)
            w_b = get_or_create_wire(b)
            gates.append((w_a, w_b))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            wires_in_use.add(w)
            return w
        else:
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
