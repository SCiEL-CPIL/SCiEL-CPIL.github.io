"""Maximum independent set in 3-uniform hypergraph of bad LCM triples."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations

    n = 128
    # Precompute pairwise LCMs
    lcm_table = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[i][j] = lcm(i + 1, j + 1)
            lcm_table[j][i] = lcm_table[i][j]

    # Build hyperedges: triples (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Use 0-indexed: vertices 0..127 represent values 1..128
    bad_hyperedges = set()
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if (lcm_table[a][b] == lcm_table[a][c] == 
                    lcm_table[b][c]):
                    bad_hyperedges.add((a, b, c))

    # For each vertex, precompute which hyperedges contain it
    vertex_to_edges = [[] for _ in range(n)]
    for (a, b, c) in bad_hyperedges:
        vertex_to_edges[a].append((a, b, c))
        vertex_to_edges[b].append((a, b, c))
        vertex_to_edges[c].append((a, b, c))

    # For each vertex, precompute a bitset of vertices that form a bad hyperedge
    # with this vertex and some other vertex
    # conflict_mask[v] = bitmask of all u such that there exists w where (v,u,w) is bad
    # More precisely: for vertex v, we need to know which pairs (u,w) are forbidden
    # when v is selected. Let's precompute for each v, the set of vertices that
    # cannot coexist with v in certain ways.
    
    # Alternative approach: for each vertex v, precompute a "forbidden pair" set
    # When v is in the solution, no pair (u,w) with u<w can both be in the solution
    # if (v,u,w) is a bad hyperedge.
    
    # For branch and bound, we'll use the following:
    # - Maintain current selected set S
    # - At each step, choose a vertex v to include or exclude
    # - When v is included, we must exclude all vertices that form bad triples
    #   with v and any already-selected vertex
    
    # Let's precompute for each vertex v and each potential "already selected" set
    # Actually, let's use a different bitset strategy:
    # For each vertex v, precompute forbid[v] = bitmask of vertices u such that
    # there exists w where (v,u,w) is a bad triple. This is a superset.
    # More precisely: if v is selected and u is selected, then no w can be selected
    # if (v,u,w) is bad.
    
    # For each pair (v,u), precompute the set of w that would create a bad triple
    # with v and u
    pair_forbidden = {}
    for (a, b, c) in bad_hyperedges:
        # For pair (a,b): c is forbidden
        # For pair (a,c): b is forbidden  
        # For pair (b,c): a is forbidden
        if (a, b) not in pair_forbidden:
            pair_forbidden[(a, b)] = 0
        if (a, c) not in pair_forbidden:
            pair_forbidden[(a, c)] = 0
        if (b, c) not in pair_forbidden:
            pair_forbidden[(b, c)] = 0
        
        pair_forbidden[(a, b)] |= (1 << c)
        pair_forbidden[(a, c)] |= (1 << b)
        pair_forbidden[(b, c)] |= (1 << a)

    # For branch and bound, we need an upper bound on how many more vertices
    # we can add. We'll use a simple greedy coloring bound or just count available.
    
    # State: selected (bitset), candidate (bitset of vertices not yet decided)
    # When we include vertex v:
    #   - Remove v from candidate
    #   - For each u in selected, remove all w in pair_forbidden.get((v,u),0) from candidate
    #   - Also remove v from candidate
    
    # Let's implement branch and bound with vertex ordering heuristic
    
    best_size = 0
    best_set = set()
    
    # Order vertices by degree (number of bad hyperedges they're in) - process high degree first
    degrees = [len(vertex_to_edges[i]) for i in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # For efficiency, we'll work with bitsets (integers)
    # full_mask = (1 << n) - 1
    
    def dfs(selected_bits, candidate_bits, current_size):
        nonlocal best_size, best_set
        
        if candidate_bits == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(i for i in range(n) if (selected_bits >> i) & 1)
            return
        
        # Upper bound: current_size + popcount(candidate_bits)
        # But we can do better with a greedy bound
        # Simple bound: current_size + number of candidates
        if current_size + bin(candidate_bits).count('1') <= best_size:
            return
        
        # Choose next vertex to branch on
        # Use first available vertex in vertex_order
        next_v = None
        for v in vertex_order:
            if (candidate_bits >> v) & 1:
                next_v = v
                break
        if next_v is None:
            if current_size > best_size:
                best_size = current_size
                best_set = set(i for i in range(n) if (selected_bits >> i) & 1)
            return
        
        # Branch 1: Include next_v
        new_selected = selected_bits | (1 << next_v)
        new_candidate = candidate_bits & ~(1 << next_v)
        
        # Remove vertices that conflict with next_v given current selected set
        # For each u in selected, remove pair_forbidden[(next_v, u)] from candidate
        # Also, any w that forms bad triple with next_v and some already selected u
        sel_list = [i for i in range(n) if (selected_bits >> i) & 1]
        to_remove = 0
        for u in sel_list:
            key1 = (min(next_v, u), max(next_v, u))
            to_remove |= pair_forbidden.get(key1, 0)
        new_candidate &= ~to_remove
        
        dfs(new_selected, new_candidate, current_size + 1)
        
        # Branch 2: Exclude next_v
        dfs(selected_bits, candidate_bits & ~(1 << next_v), current_size)
    
    full_candidate = (1 << n) - 1
    dfs(0, full_candidate, 0)
    
    result = sorted(v + 1 for v in best_set)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
