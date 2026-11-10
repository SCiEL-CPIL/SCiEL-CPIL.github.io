import itertools
import time

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    # 1. Generate seed colouring
    n = 16
    M = 17
    colours = {}
    for i in range(n):
        for j in range(i+1, n):
            colours[(i,j)] = (i+j) % M
            
    # 2. Identify tight K4s
    # A K4 is tight if it has exactly 5 distinct colours
    conflict_sets = []
    vertices = list(range(n))
    
    # Optimization: Pre-compute edge colours for faster access
    # But 16 choose 4 is 1820, which is small.
    
    for quad in itertools.combinations(vertices, 4):
        v1, v2, v3, v4 = quad
        c12 = colours[(v1,v2)]
        c13 = colours[(v1,v3)]
        c14 = colours[(v1,v4)]
        c23 = colours[(v2,v3)]
        c24 = colours[(v2,v4)]
        c34 = colours[(v3,v4)]
        
        distinct = {c12, c13, c14, c23, c24, c34}
        if len(distinct) == 5:
            conflict_sets.append(frozenset(distinct))
            
    # 3. Build conflict graph
    # Nodes 0..16
    adj = [set() for _ in range(M)]
    for cs in conflict_sets:
        # Connect all pairs in cs
        cs_list = list(cs)
        for i in range(len(cs_list)):
            for j in range(i+1, len(cs_list)):
                u = cs_list[i]
                v = cs_list[j]
                adj[u].add(v)
                adj[v].add(u)
                
    # 4. Greedy merge
    remaining = set(range(M))
    groups = []
    
    while remaining:
        # Find maximal independent set in induced subgraph on 'remaining'
        # Greedy: pick node with lowest degree, add to independent set, remove neighbors
        # Repeat until empty
        ind_set = set()
        temp_remaining = set(remaining)
        
        while temp_remaining:
            # Pick node with min degree in temp_remaining
            min_node = min(temp_remaining, key=lambda x: len(adj[x] & temp_remaining))
            ind_set.add(min_node)
            # Remove min_node and its neighbors from temp_remaining
            temp_remaining.discard(min_node)
            for neighbor in adj[min_node]:
                if neighbor in temp_remaining:
                    temp_remaining.discard(neighbor)
                    
        groups.append(ind_set)
        remaining.difference_update(ind_set)
        
    # 5. Map original colours to group indices
    colour_to_group = {}
    for idx, group in enumerate(groups):
        for c in group:
            colour_to_group[c] = idx
            
    # 6. Generate final colouring
    result = []
    for i in range(n):
        for j in range(i+1, n):
            c = colours[(i,j)]
            result.append(colour_to_group[c])
            
    # Verify validity (optional but good for safety)
    # If time permits, check if any K4 has < 5 colours
    # The greedy merge ensures no tight K4 has 2 merged colours, so it should be valid.
    # But wait, what if a K4 was NOT tight (had 6 colours) and we merged 2 colours in it?
    # It becomes 5 colours. Still valid.
    # What if we merged 3 colours in a K4 that had 6 colours?
    # It becomes 4 colours. INVALID.
    # My greedy merge only ensures that no TIGHT K4 has 2 merged colours.
    # It does NOT ensure that no NON-TIGHT K4 has 3 merged colours.
    # So I need to check validity after merging.
    
    # If invalid, we need to be more careful.
    # However, the number of groups is likely small (e.g., 5-8).
    # If a K4 had 6 colours, and we merged 3 of them, it drops to 4.
    # This is a risk.
    
    # To be safe, we can verify.
    if time.time() < deadline - 10:
        is_valid = True
        for quad in itertools.combinations(vertices, 4):
            v1, v2, v3, v4 = quad
            c12 = colour_to_group[colours[(v1,v2)]]
            c13 = colour_to_group[colours[(v1,v3)]]
            c14 = colour_to_group[colours[(v1,v4)]]
            c23 = colour_to_group[colours[(v2,v3)]]
            c24 = colour_to_group[colours[(v2,v4)]]
            c34 = colour_to_group[colours[(v3,v4)]]
            if len({c12, c13, c14, c23, c24, c34}) < 5:
                is_valid = False
                break
        if not is_valid:
            # Fallback to seed if invalid
            return [(i+j) % 17 for i in range(16) for j in range(i+1, 16)]
            
    return result
