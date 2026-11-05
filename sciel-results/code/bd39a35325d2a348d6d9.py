import random
import itertools

def construct():
    n = 20
    # 1. Create complete graph edges and proper coloring
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v, 0])
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx
            idx += 1
            
    # Proper edge coloring for K_20
    # Using the 1-factorization pattern
    center = n - 1
    colors = [0] * len(edges)
    for i in range(len(edges)):
        u, v, _ = edges[i]
        if u == center:
            other = v
        elif v == center:
            other = u
        else:
            other = None
            
        if other is not None:
            colors[i] = other % (n - 1)
        else:
            colors[i] = (u + v) % (n - 1)
            
    for i in range(len(edges)):
        edges[i][2] = colors[i]

    # 2. Build Conflict Graph
    # Node i represents edge i.
    # Edge (i, j) in conflict graph if i and j are opposite in a bad 4-cycle.
    N = len(edges)
    # adj[i] is a bitmask of vertices conflicting with i
    adj = [0] * N
    
    # Precompute 4-cycles
    # We iterate over all 4-sets of vertices
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(b, c)]
        e3 = edge_to_idx[(c, d)]
        e4 = edge_to_idx[(d, a)]
        
        # Check if bad
        if colors[e1] == colors[e3]:
            # Conflict between e1 and e3
            adj[e1] |= (1 << e3)
            adj[e3] |= (1 << e1)
        if colors[e2] == colors[e4]:
            # Conflict between e2 and e4
            adj[e2] |= (1 << e4)
            adj[e4] |= (1 << e2)
            
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(b, d)]
        e3 = edge_to_idx[(d, c)]
        e4 = edge_to_idx[(c, a)]
        
        if colors[e1] == colors[e3]:
            adj[e1] |= (1 << e3)
            adj[e3] |= (1 << e1)
        if colors[e2] == colors[e4]:
            adj[e2] |= (1 << e4)
            adj[e4] |= (1 << e2)

        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(a, c)]
        e2 = edge_to_idx[(c, b)]
        e3 = edge_to_idx[(b, d)]
        e4 = edge_to_idx[(d, a)]
        
        if colors[e1] == colors[e3]:
            adj[e1] |= (1 << e3)
            adj[e3] |= (1 << e1)
        if colors[e2] == colors[e4]:
            adj[e2] |= (1 << e4)
            adj[e4] |= (1 << e2)

    # 3. Solve Maximum Independent Set using Branch and Bound
    # We want to maximize the number of edges kept.
    # This is the size of the MIS in the conflict graph.
    
    best_size = 0
    best_mask = 0
    
    # Greedy initial solution
    def greedy_mis(adj):
        candidates = (1 << N) - 1
        result = 0
        while candidates:
            # Pick vertex with lowest degree in induced subgraph? Or just first?
            # Let's pick the one with lowest degree in candidates to keep options open?
            # Or highest degree to remove many?
            # For MIS, picking low degree is often better for size.
            # Let's just pick the first available for speed, or sort by degree.
            # Sorting is O(N log N). N=190. Fine.
            verts = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
            
            # Sort by degree in candidates
            def deg(v):
                return (adj[v] & candidates).bit_count()
            
            verts.sort(key=deg)
            
            # Pick the first one (lowest degree)
            v = verts[0]
            result |= (1 << v)
            # Remove v and its neighbors
            candidates &= ~(adj[v] | (1 << v))
            
        return result

    init_mask = greedy_mis(adj)
    best_size = init_mask.bit_count()
    best_mask = init_mask

    def dfs(candidates, current_mask, current_size):
        nonlocal best_size, best_mask
        
        if current_size + candidates.bit_count() <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Bound: Greedy coloring of the complement graph induced by candidates
        # We want to partition candidates into cliques of the original graph (conflict sets)
        # The number of cliques is an upper bound on the MIS size
        def bound(cands):
            colors = 0
            unprocessed = cands
            while unprocessed:
                # Start a new color class
                # We want to find a maximal clique in G induced by unprocessed
                # This is equivalent to a maximal independent set in G_complement induced by unprocessed
                # Let's just pick a vertex and include all its neighbors in G?
                # No, a clique in G is a set of mutually conflicting vertices.
                # In the coloring of G_complement, a color class is a clique in G.
                # So we need to find a maximal clique in G[cands].
                # Greedy clique:
                # Pick v. Include v.
                # Next vertex must be adjacent to all previously included.
                # So available = available & adj[v]
                
                # To make it a good bound, we should pick the vertex that leads to the largest clique?
                # Standard Tomita bound uses a specific ordering.
                # Simple heuristic:
                avail = unprocessed
                clique = 0
                while avail:
                    # Pick vertex with max degree in avail?
                    # Or just first?
                    # Let's pick the one with highest degree in avail to grow clique fast
                    temp = avail
                    best_v = -1
                    max_deg = -1
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        d = (adj[v] & avail).bit_count()
                        if d > max_deg:
                            max_deg = d
                            best_v = v
                        temp ^= lsb
                    
                    if best_v == -1: break
                    
                    clique |= (1 << best_v)
                    # Next candidates must be neighbors of best_v in G
                    avail = avail & adj[best_v]
                    
                colors += 1
                unprocessed = unprocessed & ~clique
                
            return colors

        ub = bound(candidates)
        if current_size + ub <= best_size:
            return
            
        # Branch
        # Pick a vertex to branch on
        # Heuristic: Pick vertex with highest degree in candidates?
        temp = candidates
        best_v = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            d = (adj[v] & candidates).bit_count()
            if d > max_deg:
                max_deg = d
                best_v = v
            temp ^= lsb
            
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New candidates: candidates minus v and its neighbors
        new_cands = candidates & ~(adj[v] | v_bit)
        dfs(new_cands, current_mask | v_bit, current_size + 1)
        
        # Branch 2: Exclude v
        # New candidates: candidates minus v
        new_cands = candidates & ~v_bit
        dfs(new_cands, current_mask, current_size)

    all_vertices = (1 << N) - 1
    dfs(all_vertices, 0, 0)
    
    # Construct result
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(edges[i])
            
    return result
