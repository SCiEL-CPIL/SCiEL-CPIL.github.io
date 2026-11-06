import random

def construct():
    n = 20
    # Initial graph: 4 disjoint K5s
    adj = [set() for _ in range(n)]
    for i in range(4):
        start = i * 5
        for u in range(start, start + 5):
            for v in range(start, start + 5):
                if u != v:
                    adj[u].add(v)
                    adj[v].add(u)
    
    def is_k33_free(adj):
        # Check if graph has K3,3
        # For each pair of vertices (u, v), find common neighbors
        # If there are 3 common neighbors, and those 3 share 3 common neighbors with u,v?
        # No, K3,3 means there exist A, B with |A|=3, |B|=3 such that all cross edges exist.
        # We can check all subsets of size 3 for A, and see if their common neighborhood has size >= 3.
        # Number of subsets C(20,3) = 1140.
        # For each A, compute common neighbors.
        # If |CommonNeigh(A)| >= 3, then there is a K3,3?
        # Yes, if A={a,b,c} and N(a)∩N(b)∩N(c) contains {x,y,z}, then A and {x,y,z} form K3,3.
        # So we just need to check if any 3 vertices have >= 3 common neighbors.
        
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                # For each k in common_ij? No.
                # We need to check triples (i,j,k).
                # It is faster to iterate triples.
                pass
        
        # Optimized check:
        # Precompute common neighbors for all pairs?
        # C(20,2) = 190 pairs.
        # For each pair (u,v), let C_uv = N(u) ∩ N(v).
        # A K3,3 exists if there is a triple (u,v,w) such that |N(u) ∩ N(v) ∩ N(w)| >= 3.
        # This is equivalent to: there exist u,v,w such that C_uv contains w? No.
        # C_uv is the set of common neighbors of u and v.
        # If w is in C_uv, then u,v,w share at least 1 common neighbor.
        # We need 3 common neighbors for u,v,w.
        # So we need |N(u) ∩ N(v) ∩ N(w)| >= 3.
        
        # Iterate all triples (u,v,w)
        for u in range(n):
            for v in range(u+1, n):
                common_uv = adj[u] & adj[v]
                # We need to find w such that |common_uv ∩ N(w)| >= 3.
                # This means w must be connected to at least 3 vertices in common_uv.
                # So for each w not in {u,v}, check if |adj[w] & common_uv| >= 3.
                # Note: w can be in common_uv? No, w is a vertex.
                # If w is in common_uv, then w is a neighbor of u and v.
                # Then N(w) ∩ N(u) ∩ N(v) = N(w) ∩ common_uv.
                # We need this intersection to have size >= 3.
                
                # To speed up, we can just check all w.
                # But we can restrict w to vertices connected to u and v?
                # If w is not connected to u, then |N(w) ∩ N(u) ∩ N(v)| = 0.
                # So w MUST be in N(u) ∩ N(v) = common_uv.
                # So we only need to check w in common_uv.
                
                for w in common_uv:
                    # Check if w has at least 3 neighbors in common_uv
                    # Note: w is in common_uv, so w is a neighbor of u and v.
                    # The common neighbors of u,v,w are common_uv ∩ N(w).
                    # Since w is in common_uv, w is a neighbor of u and v.
                    # But does w connect to itself? No.
                    # So we check |adj[w] & common_uv|.
                    # Note: adj[w] does not contain w.
                    # So this counts neighbors of w that are also neighbors of u and v.
                    if len(adj[w] & common_uv) >= 3:
                        return False
        return True

    def try_add_edge(u, v):
        if v in adj[u]:
            return False
        # Check if adding (u,v) creates K3,3
        # New graph G' = G + (u,v)
        # We only need to check K3,3 that use the new edge (u,v).
        # As derived before:
        # Case 1: u in A, v in B.
        # Need a,b in A\{u}, c,d in B\{v}.
        # Edges: u-c, u-d, v-a, v-b, a-c, a-d, b-c, b-d.
        # c,d must be in N(u).
        # a,b must be in N(v).
        # a,b,c,d must be such that a,b connected to c,d.
        
        # Let C = N(u) \ {v}
        # Let D = N(v) \ {u}
        # We need c,d in C and a,b in D such that:
        # a,b in D and a,b connected to c,d.
        # i.e., N(a) ∩ N(b) ∩ C contains {c,d}?
        # Actually, we just need to find ANY such a,b,c,d.
        
        # Efficient check:
        # Iterate over pairs c,d in C.
        # For each pair (c,d), find a,b in D such that a,b are connected to c and d.
        # i.e., a,b in N(c) ∩ N(d) ∩ D.
        # If |N(c) ∩ N(d) ∩ D| >= 2, then we have a,b.
        
        C = adj[u] - {v}
        D = adj[v] - {u}
        
        for c in C:
            for d in C:
                if c == d: continue
                # Find common neighbors of c and d that are in D
                common_cd_in_D = (adj[c] & adj[d]) & D
                if len(common_cd_in_D) >= 2:
                    return False
        
        return True

    # Local Search
    best_edges = len([1 for u in range(n) for v in adj[u]]) // 2
    best_adj = [set(s) for s in adj]
    
    for _ in range(100): # Iterations
        changed = True
        while changed:
            changed = False
            # Try to add any edge
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        if try_add_edge(u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            changed = True
                            # Restart inner loop to check new possibilities
                            break
                if changed:
                    break
        
        current_edges = len([1 for u in range(n) for v in adj[u]]) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [set(s) for s in adj]
            
        # Perturbation: Remove a random edge to escape local optimum
        if best_edges >= current_edges:
            edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if edges_list:
                ru, rv = random.choice(edges_list)
                adj[ru].remove(rv)
                adj[rv].remove(ru)
            # Also try randomizing initial state slightly?
            # Or just continue with the perturbed graph

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
