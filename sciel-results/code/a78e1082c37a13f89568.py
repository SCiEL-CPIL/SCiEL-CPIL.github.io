"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 512] such that lcm(a, b) <= 512 for every pair (a, b).
    
    Approach:
    1. Model as Maximum Independent Set (MIS) on a graph where edges represent conflicts (lcm > 512).
    2. Use a Branch and Bound algorithm with bitset representations for efficiency.
    3. Prune branches where the theoretical maximum size (current_size + remaining_count) cannot exceed the best found.
    4. Initial solution from Greedy Min-Degree algorithm.
    """
    import math
    import time
    
    N = 512
    
    # Build conflict graph using bitsets
    # conflicts[i] is a bitmask where bit j is 1 if i and j conflict
    # We use 0-indexed: node i represents number i+1
    conflicts = [0] * N
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = math.gcd(a, b)
            lcm = a * b // g
            if lcm > N:
                # Mark conflict
                conflicts[a-1] |= (1 << (b-1))
                conflicts[b-1] |= (1 << (a-1))
                
    best_set = []
    best_size = 0
    
    # Greedy initialization: pick min degree node
    remaining_mask = (1 << N) - 1
    current_set = []
    
    temp_remaining = remaining_mask
    while temp_remaining:
        # Find node with min degree in temp_remaining
        min_deg = N + 1
        best_node = -1
        m = temp_remaining
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            neighbors = conflicts[idx] & temp_remaining
            deg = neighbors.bit_count()
            if deg < min_deg:
                min_deg = deg
                best_node = idx
            m ^= lsb
            
        if best_node == -1:
            break
            
        current_set.append(best_node + 1)
        # Remove best_node and its neighbors
        temp_remaining &= ~(1 << best_node)
        temp_remaining &= ~conflicts[best_node]
        
    best_size = len(current_set)
    best_set = current_set[:]

    start_time = time.time()
    TIME_LIMIT = 55.0 # Leave 5s for overhead
    
    def search(remaining_mask, current_size, current_nodes_list):
        nonlocal best_size, best_set
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # Pruning
        # Upper bound: current_size + number of remaining nodes
        if current_size + remaining_mask.bit_count() <= best_size:
            return
            
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_nodes_list[:]
            return
            
        # Select a node to branch on
        # Heuristic: Pick node with highest degree in remaining graph
        max_deg = -1
        branch_node = -1
        
        m = remaining_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (conflicts[idx] & remaining_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_node = idx
            m ^= lsb
            
        if branch_node == -1:
            return
            
        # Branch 1: Include branch_node
        new_remaining_include = remaining_mask & ~(1 << branch_node)
        new_remaining_include &= ~conflicts[branch_node]
        
        # Branch 2: Exclude branch_node
        new_remaining_exclude = remaining_mask & ~(1 << branch_node)
        
        # Recurse
        # Explore Include branch first
        current_nodes_list.append(branch_node + 1)
        search(new_remaining_include, current_size + 1, current_nodes_list)
        current_nodes_list.pop()
        
        search(new_remaining_exclude, current_size, current_nodes_list)
        
    # Start search
    search(remaining_mask, 0, [])
    
    return best_set

# EVOLVE-BLOCK-END
