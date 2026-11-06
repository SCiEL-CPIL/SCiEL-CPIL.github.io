import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    # 1. Precompute LCMs
    # lcm(i, j) for 1 <= i < j <= N
    # We only need to check equality, so we can store the values.
    # Max LCM of two numbers <= 128 is 128*127 = 16256. Small.
    
    lcm_val = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_val[i][j] = (i * j) // gcd(i, j)
    
    # 2. Identify Bad Triples
    # Triple (a, b, c) with a < b < c is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # w_set[v][u] is a bitmask of w such that {v, u, w} is a bad triple.
    # Indices 1..N. We use 0-indexed bits for 1..N? 
    # Let's use bit i-1 for vertex i.
    w_set = [[0] * (N + 1) for _ in range(N + 1)]
    
    # To find bad triples efficiently:
    # For each pair (a, b), let L = lcm(a, b).
    # We need c such that lcm(a, c) == L and lcm(b, c) == L.
    # Note: c must be distinct from a, b.
    # Also, we should only add (a, b, c) if a < b < c to avoid duplicates.
    
    # Iterate a, b
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L = lcm_val[a][b]
            # Find c
            # c must be such that lcm(a, c) == L and lcm(b, c) == L.
            # This implies c divides L.
            # Also lcm(a, c) == L means that for every prime power p^k in L, 
            # if a has p^i, c must have p^j such that max(i, j) = k.
            # Simpler: Just check all c > b?
            # Since N is small, checking all c in 1..N is fine.
            for c in range(b + 1, N + 1):
                if lcm_val[a][c] == L and lcm_val[b][c] == L:
                    bad_triples.append((a, b, c))
                    # Update w_set
                    # {a, b, c}
                    w_set[a][b] |= (1 << (c - 1))
                    w_set[b][a] |= (1 << (c - 1))
                    w_set[a][c] |= (1 << (b - 1))
                    w_set[c][a] |= (1 << (b - 1))
                    w_set[b][c] |= (1 << (a - 1))
                    w_set[c][b] |= (1 << (a - 1))

    # 3. Greedy Initial Solution
    # Sort vertices by "badness" (degree in hypergraph) descending?
    # Or just iterate 1..N?
    # Let's try a greedy that picks vertex with fewest conflicts first?
    # Or most conflicts?
    # Let's try a simple greedy: iterate 1..N, add if valid.
    # To improve, we can try multiple orderings or a heuristic.
    # Given the B&B will find the optimum, the initial bound just needs to be decent.
    
    def popcount(x):
        return bin(x).count("1")
        
    # Heuristic: Vertices involved in more bad triples are "harder" to place.
    # Let's compute degree.
    degree = [0] * (N + 1)
    for a, b, c in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        
    # Try a few greedy orderings to get a good lower bound
    best_greedy_size = 0
    best_greedy_set = 0
    
    # Ordering 1: Natural
    # Ordering 2: High degree first
    # Ordering 3: Low degree first
    
    def run_greedy(order):
        selected_mask = 0
        selected_count = 0
        # We need to check validity against selected set.
        # For each v in order:
        #   Check if v is compatible with selected_mask.
        #   If yes, add to selected_mask.
        
        for v in order:
            # Check compatibility with selected_mask
            # v is compatible if there are no a, b in selected_mask such that {a, b, v} is bad.
            # This is: For all a in selected_mask, (selected_mask & w_set[v][a]) == 0.
            compatible = True
            m = selected_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                a = lsb.bit_length() # This gives index of highest bit? No.
                # bit_length of 1<<k is k+1. So index is k.
                # If lsb is 1 << (a-1), then a-1 is the index.
                # lsb.bit_length() - 1 gives the index of the bit.
                # So vertex id = bit_length.
                a_id = lsb.bit_length()
                
                if selected_mask & w_set[v][a_id]:
                    compatible = False
                    break
                m ^= lsb
            
            if compatible:
                selected_mask |= (1 << (v - 1))
                selected_count += 1
                
        return selected_count, selected_mask

    # Order 1: 1..N
    size1, mask1 = run_greedy(range(1, N + 1))
    # Order 2: Sorted by degree descending
    order2 = sorted(range(1, N + 1), key=lambda x: -degree[x])
    size2, mask2 = run_greedy(order2)
    # Order 3: Sorted by degree ascending
    order3 = sorted(range(1, N + 1), key=lambda x: degree[x])
    size3, mask3 = run_greedy(order3)
    
    best_greedy_size = max(size1, size2, size3)
    if size1 >= size2 and size1 >= size3:
        best_greedy_set = mask1
    elif size2 >= size1 and size2 >= size3:
        best_greedy_set = mask2
    else:
        best_greedy_set = mask3

    best_size = best_greedy_size
    best_mask = best_greedy_set

    # 4. Branch and Bound
    # State: selected_mask, available_mask
    # We use recursion.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute bit masks for vertices for fast access?
    # vertex_bit[v] = 1 << (v-1)
    vertex_bit = [0] * (N + 1)
    for v in range(1, N + 1):
        vertex_bit[v] = 1 << (v - 1)

    # List of bad triples for matching bound
    # We can use the list directly.
    
    def greedy_matching_bound(available_mask):
        local_avail = available_mask
        matches = 0
        for a, b, c in bad_triples:
            # Check if a, b, c are in local_avail
            if (local_avail & vertex_bit[a]) and (local_avail & vertex_bit[b]) and (local_avail & vertex_bit[c]):
                matches += 1
                local_avail &= ~vertex_bit[a]
                local_avail &= ~vertex_bit[b]
                local_avail &= ~vertex_bit[c]
                if local_avail == 0:
                    break
        return matches

    def is_compatible(v, selected_mask):
        # Check if v can be added to selected_mask
        # i.e., no pair in selected_mask forms a bad triple with v
        m = selected_mask
        while m:
            lsb = m & -m
            a_id = lsb.bit_length()
            if selected_mask & w_set[v][a_id]:
                return False
            m ^= lsb
        return True

    def search(selected_mask, available_mask, current_size):
        nonlocal best_size, best_mask
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        avail_count = popcount(available_mask)
        
        if current_size + avail_count <= best_size:
            return
            
        # Calculate tighter bound
        match_count = greedy_matching_bound(available_mask)
        ub = current_size + avail_count - match_count
        if ub <= best_size:
            return
            
        if avail_count == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        # Choose vertex to branch on
        # Heuristic: Pick vertex with highest degree in the available subgraph?
        # Or just first available.
        # Let's pick the one with highest global degree for now.
        # To be faster, we can just pick the first available vertex.
        # Given the small N, this should be fine.
        
        # Let's try picking the vertex that is part of the most bad triples 
        # involving other available vertices.
        # This is expensive. Let's stick to global degree or first.
        # Let's use first available for speed.
        
        # Get first available vertex
        lsb = available_mask & -available_mask
        v = lsb.bit_length()
        
        # Branch 1: Include v
        if is_compatible(v, selected_mask):
            new_selected = selected_mask | vertex_bit[v]
            new_available = available_mask & ~vertex_bit[v]
            
            # Remove u from new_available if {u, v, w} is bad for some w in selected_mask
            # For each u in new_available:
            #   if (selected_mask & w_set[v][u]) != 0: remove u
            
            # Optimize:
            # We can iterate through bits in new_available.
            # Or, we can compute the union of w_set[v][u] for all u in new_available?
            # No, we need to remove u if there exists w in S.
            # Condition: selected_mask & w_set[v][u] != 0.
            
            # Let's iterate u in new_available.
            m = new_available
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length()
                if selected_mask & w_set[v][u]:
                    new_available &= ~lsb_u
                m ^= lsb_u
                
            search(new_selected, new_available, current_size + 1)
            
        # Branch 2: Exclude v
        search(selected_mask, available_mask & ~vertex_bit[v], current_size)
        
    # Start search
    # Initial available is all vertices
    initial_available = (1 << N) - 1
    search(0, initial_available, 0)
    
    # Return the set as a list of integers
    result_set = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length()
        result_set.append(v)
        m ^= lsb
        
    return result_set

# EVOLVE-BLOCK-END
