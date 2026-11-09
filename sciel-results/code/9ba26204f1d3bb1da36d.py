import time
import sys

# Set recursion limit high just in case, though iterative or deep recursion might be needed
sys.setrecursionlimit(3000)

N = 256
# Precompute conflicts
# conflict[i] is a bitmask of j such that (i+j) divides i*j
# i, j in 1..256
# Bit i corresponds to number i.
# We use 0-indexed bits for numbers 1..256?
# Let's map number k to bit (k-1).
# So number 1 is bit 0, number 256 is bit 255.

conflict = [0] * (N + 1)

def compute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                # Set bit for b in conflict[a] and a in conflict[b]
                # Bit index is value - 1
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))

compute_conflicts()

def popcount(x):
    return bin(x).count('1')

def color_bound(mask):
    """
    Returns the number of colors in a greedy coloring of the subgraph induced by 'mask'.
    This is an upper bound on the size of the maximum independent set in that subgraph.
    """
    if mask == 0:
        return 0
    colors = 0
    uncolored = mask
    while uncolored:
        colors += 1
        # Greedily select an independent set for this color
        # We pick vertices one by one from 'uncolored' that are not adjacent to previously picked ones in this color
        available = uncolored
        current_color_set = 0
        while available:
            # Pick the lowest set bit for determinism and speed
            # This is a simple heuristic. Better heuristics might exist but this is fast.
            lsb = available & -available
            v_idx = lsb.bit_length() - 1
            v_val = v_idx + 1 # Convert back to 1-based index for conflict lookup
            
            current_color_set |= lsb
            # Remove v and its neighbors from available
            # conflict[v_val] has bits for neighbors
            available &= ~(conflict[v_val] | lsb)
        
        # Remove this color set from uncolored
        uncolored &= ~current_color_set
        
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    best_size = 0
    best_mask = 0
    
    # Initial candidates: all vertices 1..256
    # Bit 0 is 1, Bit 255 is 256
    initial_mask = (1 << N) - 1
    
    # Heuristic: Order vertices by degree?
    # For the initial call, we can just pass the full mask.
    # The search function will pick vertices dynamically.
    
    def search(candidates_mask, current_mask, current_size):
        nonlocal best_size, best_mask
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning
        # Upper bound using coloring
        # If current_size + color_bound(candidates_mask) <= best_size, prune
        # Optimization: If popcount(candidates_mask) + current_size <= best_size, prune immediately (cheaper)
        if current_size + popcount(candidates_mask) <= best_size:
            return
            
        # If the coloring bound is tighter, use it.
        # Coloring is O(N^2) worst case, but usually fast.
        # Let's try to use it if the gap is small or always?
        # For N=256, it's likely fast enough to call at each node.
        cb = color_bound(candidates_mask)
        if current_size + cb <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Heuristic to pick a vertex v
        # We want to pick a vertex that is "central" to prune more.
        # High degree in the remaining subgraph is a good indicator.
        # Let's find the vertex in candidates_mask with the max degree within candidates_mask.
        max_deg = -1
        v_idx = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            val = idx + 1
            # Degree of val in candidates_mask
            # neighbors = conflict[val] & candidates_mask
            deg = popcount(conflict[val] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                v_idx = idx
            temp_mask ^= lsb
            
        if v_idx == -1:
             return # Should not happen if candidates_mask != 0

        v_bit = 1 << v_idx
        v_val = v_idx + 1
        
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_candidates = candidates_mask & ~(conflict[v_val] | v_bit)
        search(new_candidates, current_mask | v_bit, current_size + 1)
        
        # Check time again before second branch
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~v_bit
        search(new_candidates, current_mask, current_size)

    search(initial_mask, 0, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    # Fallback if something went wrong or no set found (should not happen, single element is always valid)
    if not result:
        return [1]
        
    return result
