"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s wall-clock limit
    
    # 1. Pre-filter candidates:
    #    If x <= 128, then for any y, lcm(x,y) is a multiple of x.
    #    If y is a multiple of x, lcm(x,y) = y. If y <= 256, lcm <= 256 (Fail).
    #    If x <= 128, there exists a multiple of x in [2, 256] (specifically x*2 if x<=128, or x itself if we considered 1 but 1 is excluded).
    #    Actually, if x <= 128, 2x is in [4, 256]. lcm(x, 2x) = 2x <= 256.
    #    So any x <= 128 is incompatible with 2x.
    #    Wait, the constraint is for ALL distinct pairs in the SET.
    #    If we include x <= 128, we MUST exclude all multiples of x that are <= 256.
    #    This makes small numbers very "expensive" in terms of conflicts.
    #    However, small numbers have high reciprocal weight.
    #    Let's check if x <= 128 can be in an optimal set.
    #    If x is in S, then no multiple of x (other than x) can be in S.
    #    Specifically, 2x, 3x, ... <= 256 cannot be in S.
    #    The weight of x is 1/x. The "cost" is removing 1/(2x), 1/(3x), etc.
    #    Usually, 1/x > 1/(2x) + 1/(3x) + ... is false for small x?
    #    1/2 = 0.5. 1/4 + 1/6 + 1/8 + ... = 0.25 + 0.166 + 0.125 ... sum is roughly 0.5-0.6.
    #    It's a trade-off. But we can just include all 2..256 in the candidate pool and let the solver decide.
    #    However, to speed up, we can note that if x > 128, it has no multiples in [2, 256] (except itself, which is not distinct).
    #    So x > 128 only conflicts with y if lcm(x,y) <= 256.
    #    Since x > 128, y must be small to make lcm small?
    #    lcm(x,y) = x*y/gcd(x,y). If x > 128, and y >= 2.
    #    If gcd(x,y) = 1, lcm = xy > 256 (since x>128, y>=2 => xy > 256).
    #    So if x > 128, it is compatible with all y that are coprime to x.
    #    It is incompatible with y if gcd(x,y) = d such that x*y/d <= 256.
    #    This implies y <= 256*d/x. Since x>128, 256/x < 2. So y <= 2*d.
    #    So large numbers only conflict with small numbers sharing large factors.
    
    # Candidate set: [2, 256]
    # Weights: 1/x
    # Conflict: lcm(x,y) <= 256
    
    # To make B&B fast, we should sort candidates by weight descending (smallest x first).
    # But small x have many conflicts.
    # Let's just use the full set [2, 256] but order them by weight descending for the B&B.
    # Actually, B&B on 255 nodes is still feasible if the graph is sparse or structured well.
    # The graph is NOT sparse for small nodes.
    # Node 2 conflicts with all even numbers.
    # Node 3 conflicts with all multiples of 3.
    
    # Strategy:
    # 1. Build conflict matrix (bitset) for [2, 256].
    # 2. Sort nodes by weight descending (2, 3, 4, ...).
    # 3. Use recursive B&B:
    #    - State: current set S, available candidates A (bitset), current sum.
    #    - Bound: current sum + sum of weights of all nodes in A (optimistic: assume no conflicts).
    #    - If bound <= best_sum, prune.
    #    - Pick a node v from A (e.g., the one with highest weight, or highest degree).
    #    - Branch 1: Include v. New A = A & ~neighbors[v] & ~bit(v).
    #    - Branch 2: Exclude v. New A = A & ~bit(v).
    
    # Precompute neighbors as bitsets.
    # Python ints are good for bitsets.
    
    N = 256
    # Map value to index
    # Values are 2..256
    # Index i corresponds to value i+2
    # Total nodes = 255
    
    # Let's use a list of values
    values = list(range(2, 257))
    n_nodes = len(values)
    
    # weights
    weights = [1.0 / v for v in values]
    
    # Conflict matrix (list of bitmasks)
    # neighbors[i] is a bitmask where bit j is set if values[i] and values[j] conflict
    neighbors = [0] * n_nodes
    
    # Optimization: Precompute lcm conflicts
    # For each i, j > i:
    # if lcm(values[i], values[j]) <= 256:
    #    neighbors[i] |= 1 << j
    #    neighbors[j] |= 1 << i
    
    # This double loop is 255^2 / 2 ~ 32k iterations. Fast.
    for i in range(n_nodes):
        vi = values[i]
        for j in range(i + 1, n_nodes):
            vj = values[j]
            # Calculate LCM
            # lcm(a,b) = a*b//gcd(a,b)
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
    
    # Sort nodes by weight descending?
    # B&B is more effective if we branch on nodes with high impact.
    # Standard MWIS B&B usually orders by degree or weight.
    # Let's reorder the nodes to put high weight (small value) first?
    # Or high degree (many conflicts) first?
    # If we branch on a high-degree node, we prune many branches quickly.
    # Let's sort by degree descending, then weight descending.
    
    degrees = [bin(neighbors[i]).count('1') for i in range(n_nodes)]
    
    # Create sorted indices
    # Key: (-degree, -weight)
    sorted_indices = sorted(range(n_nodes), key=lambda i: (-degrees[i], -weights[i]))
    
    # Remap neighbors to new order
    # new_neighbors[i] corresponds to old node sorted_indices[i]
    # We need to map old bit positions to new bit positions.
    # old_pos -> new_pos
    pos_map = [0] * n_nodes
    for new_pos, old_pos in enumerate(sorted_indices):
        pos_map[old_pos] = new_pos
    
    new_neighbors = [0] * n_nodes
    new_weights = [0.0] * n_nodes
    
    for new_i in range(n_nodes):
        old_i = sorted_indices[new_i]
        new_weights[new_i] = weights[old_i]
        mask = neighbors[old_i]
        new_mask = 0
        # Iterate set bits in mask
        # m = mask
        # while m:
        #   lowbit = m & -m
        #   old_j = lowbit.bit_length() - 1
        #   new_j = pos_map[old_j]
        #   new_mask |= (1 << new_j)
        #   m ^= lowbit
        m = mask
        while m:
            lowbit = m & -m
            old_j = lowbit.bit_length() - 1
            new_j = pos_map[old_j]
            new_mask |= (1 << new_j)
            m ^= lowbit
        new_neighbors[new_i] = new_mask
    
    # Precompute suffix sums of weights for bounding
    # suffix_sum[i] = sum(new_weights[i:])
    suffix_sum = [0.0] * (n_nodes + 1)
    for i in range(n_nodes - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + new_weights[i]
    
    best_sum = 0.0
    best_set = []  # List of original values
    
    # Global time check
    def time_check():
        return time.time() > deadline
    
    # Recursive B&B
    # available_mask: bitmask of available nodes
    # current_sum: sum of weights in current set
    # current_set: list of indices in current set
    
    # To speed up, we can use a greedy bound:
    # Instead of sum of all available, we can try to greedily select non-conflicting nodes
    # from available to get a tighter upper bound.
    # But that's expensive. Simple sum is O(1).
    # Let's stick to simple sum first. If it's too slow, we'll see.
    
    # One optimization: If available_mask is 0, update best.
    
    def solve(available_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time_check():
            return
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
        
        # Bound
        # Find the lowest set bit in available_mask to start the suffix sum?
        # No, suffix_sum is based on index order.
        # The available nodes are a subset.
        # Upper bound = current_sum + sum(weights[i] for i in available_bits)
        # We can precompute a lookup or just iterate?
        # Iterating 255 bits is fast.
        # Or, we can use the fact that we process nodes in order.
        # If we always branch on the lowest index in available_mask?
        # Let's try that. It simplifies the bound calculation if we maintain the state differently.
        
        # Alternative B&B formulation:
        # Process nodes in sorted order (0 to n-1).
        # State: index i, available_mask (only bits >= i can be set?), current_sum.
        # This is tricky with bitmasks if we skip nodes.
        
        # Let's stick to the "pick a node from available" approach.
        # To make bound calculation fast, let's assume we pick the node with the LOWEST INDEX in available_mask.
        # Why? Because suffix_sum[i] is precomputed.
        # If we pick node v = lowest set bit index in available_mask.
        # Then the max possible sum from remaining available nodes is bounded by suffix_sum[v].
        # Is this valid?
        # suffix_sum[v] = sum of weights of ALL nodes from v to end.
        # The available nodes are a subset of nodes from v to end (since v is the lowest index).
        # So sum(weights of available) <= suffix_sum[v].
        # Yes, this is a valid upper bound.
        
        # Find lowest set bit
        lowbit = available_mask & -available_mask
        v = lowbit.bit_length() - 1
        
        # Bound check
        if current_sum + suffix_sum[v] <= best_sum:
            return
        
        # Branch 1: Include v
        # New available = available_mask & ~neighbors[v] & ~lowbit
        new_mask_incl = available_mask & ~new_neighbors[v] & ~lowbit
        solve(new_mask_incl, current_sum + new_weights[v], current_set + [v])
        
        # Branch 2: Exclude v
        # New available = available_mask & ~lowbit
        new_mask_excl = available_mask & ~lowbit
        solve(new_mask_excl, current_sum, current_set)
        
    # Initial call
    # All nodes available
    init_mask = (1 << n_nodes) - 1
    solve(init_mask, 0.0, [])
    
    # Map best_set (indices in sorted order) back to original values
    result_values = []
    for idx in best_set:
        old_idx = sorted_indices[idx]
        result_values.append(values[old_idx])
    
    # Fallback: if best_set is empty (should not happen, at least one node can be picked),
    # return a single largest number (highest reciprocal? No, smallest number 2? No, 2 conflicts with many.
    # Actually, any single number is a valid set.
    # The greedy seed found a set. My B&B should find at least as good.
    if not result_values:
        # Just return [256] as a safe non-empty set
        result_values = [256]
        
    return result_values

# EVOLVE-BLOCK-END
