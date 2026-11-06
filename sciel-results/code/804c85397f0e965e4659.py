import random
import time

def construct():
    n = 20
    # Use bitmasks for adjacency for speed
    # adj[i] is an integer bitmask of neighbors of i
    
    def popcount(x):
        return bin(x).count('1')

    def is_k33_free(adj):
        # Check all triples (u, v, w)
        # If they have >= 3 common neighbors, K3,3 exists.
        # Common neighbors of u, v, w: adj[u] & adj[v] & adj[w]
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = adj[u] & adj[v]
                if popcount(common_uv) < 3:
                    continue
                # Iterate w in common_uv? No, w can be any vertex.
                # But if w is not in N(u) or N(v), intersection is 0.
                # So w must be in common_uv? 
                # Wait. If w is NOT in N(u), then adj[w] & adj[u] = 0.
                # So w MUST be in N(u) AND N(v).
                # So w is in common_uv.
                # We need popcount(adj[u] & adj[v] & adj[w]) >= 3.
                # Note: adj[w] does not contain w.
                # So we check for w in common_uv:
                # if popcount(common_uv & adj[w]) >= 3: return False
                # Optimization: If popcount(common_uv) < 3, skip.
                # If w is in common_uv, then w is a neighbor of u and v.
                # The common neighbors of u,v,w are common_uv \ {w} intersected with adj[w]?
                # No. Common neighbors of u,v,w are vertices x such that x~u, x~v, x~w.
                # So x is in common_uv AND x is in adj[w].
                # So count = popcount(common_uv & adj[w]).
                # We need this count >= 3.
                
                # We can iterate w in common_uv.
                # To iterate bits:
                mask = common_uv
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    mask ^= lsb
                    if popcount(common_uv & adj[w]) >= 3:
                        return False
        return True

    def try_add_edge(u, v, adj):
        # Check if adding (u,v) creates K3,3
        # New edge (u,v).
        # Any K3,3 using (u,v) must have u and v in opposite partitions?
        # Or same? K3,3 is bipartite.
        # If u and v are in the same partition A, then they are not connected in K3,3.
        # So (u,v) must be a cross edge.
        # So u in A, v in B.
        # The other 2 in A are a,b. The other 2 in B are c,d.
        # Edges: u-c, u-d, v-a, v-b, a-c, a-d, b-c, b-d.
        # Also u-v is the new edge.
        # We need to find a,b in N(v) \ {u} and c,d in N(u) \ {v}
        # such that a,b are connected to c,d.
        
        C = adj[u] & ~(1 << v)
        D = adj[v] & ~(1 << u)
        
        # We need a pair c,d in C and a pair a,b in D
        # such that a,b in N(c) and a,b in N(d).
        # i.e., N(c) & N(d) & D has size >= 2.
        
        # Iterate pairs in C?
        # C has size deg(u).
        # If deg(u) is large, this is slow.
        # But n=20, so max deg is 19.
        # Pairs in C: ~190.
        # For each pair, check intersection.
        
        mask_c = C
        while mask_c:
            lsb_c = mask_c & -mask_c
            c = lsb_c.bit_length() - 1
            mask_c ^= lsb_c
            
            # For each d in C, d != c
            mask_d = C & ~(1 << c)
            while mask_d:
                lsb_d = mask_d & -mask_d
                d = lsb_d.bit_length() - 1
                mask_d ^= lsb_d
                
                # Check if N(c) & N(d) & D has >= 2 bits
                common_cd_in_D = adj[c] & adj[d] & D
                if popcount(common_cd_in_D) >= 2:
                    return False
        return True

    # Initialization: Random graph with p=0.5, then clean up
    adj = [0] * n
    random.seed(42)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Clean up: Remove edges that create K3,3
    # Greedy removal
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    # Check if removing (i,j) is necessary?
                    # Just check if current graph is valid.
                    # If not, remove a random edge involved in a violation?
                    # Simpler: If is_k33_free is False, find a violation and remove an edge.
                    pass
        
        if not is_k33_free(adj):
            # Find a K3,3 and remove an edge
            # We need to find the violation.
            # Reuse the logic from is_k33_free but return the edge to remove.
            for u in range(n):
                found = False
                for v in range(u + 1, n):
                    common_uv = adj[u] & adj[v]
                    if popcount(common_uv) < 3:
                        continue
                    mask_w = common_uv
                    while mask_w:
                        lsb_w = mask_w & -mask_w
                        w = lsb_w.bit_length() - 1
                        mask_w ^= lsb_w
                        if popcount(common_uv & adj[w]) >= 3:
                            # Found K3,3 involving u,v,w
                            # Remove edge (u,v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    best_adj = [x for x in adj]
    best_edges = sum(popcount(x) for x in adj) // 2

    # Simulated Annealing / Iterated Local Search
    T = 10.0
    start_time = time.time()
    
    while time.time() - start_time < 110:
        # Try to add edges
        added = False
        for _ in range(100): # Limit attempts per phase
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            if (adj[u] >> v) & 1: continue
            
            if try_add_edge(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                current_edges = sum(popcount(x) for x in adj) // 2
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [x for x in adj]
        
        if not added:
            # Stuck. Remove an edge to escape.
            # Remove a random edge, or an edge that is "critical"
            # Random removal is simple.
            edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append((i, j))
                    mask ^= lsb
            
            if edges:
                u, v = random.choice(edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Cool down?
        T *= 0.95

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lsb
            
    return edges
