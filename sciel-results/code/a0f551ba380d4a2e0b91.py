import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # 1. Construct Polarity Graph of PG(2,6)
    # Points are 1D subspaces of GF(7)^3
    # We represent points as tuples (x, y, 1) for affine part, and (x, y, 0) for line at infinity?
    # Standard representation:
    # Points in PG(2, q) are equivalence classes of non-zero vectors in GF(q)^3.
    # We can list them by:
    # 1. (1, y, z) for y, z in GF(q) -> q^2 points
    # 2. (x, 1, z) for x, z in GF(q) -> q points (excluding (0,1,z) which is in set 1? No)
    # Let's use a standard enumeration.
    # Total q^2 + q + 1 points.
    
    q = 6
    p = 7 # Field size
    
    points = []
    # Canonical representatives
    # For each vector v in GF(p)^3 \ {0}, normalize it.
    # Normalization: Let k be the first non-zero component. Divide all components by v[k].
    # This gives a unique representative for each 1D subspace.
    
    seen = set()
    for x in range(p):
        for y in range(p):
            for z in range(p):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    rep = (1, (y * pow(x, p-2, p)) % p, (z * pow(x, p-2, p)) % p)
                elif y != 0:
                    rep = (0, 1, (z * pow(y, p-2, p)) % p)
                else:
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    n = len(points) # Should be 43
    
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Define Polarity
    # Symmetric matrix A over GF(7)
    # We need a non-degenerate symmetric matrix.
    # A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] is the standard dot product.
    # This defines the polarity x . y = 0.
    # Absolute points are those where x . x = 0.
    
    A = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    
    def dot(u, v):
        return (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % p
    
    # Build adjacency list for G (Polarity Graph)
    # u ~ v if dot(u, v) == 0 and u != v
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Build complement graph adj_comp
    # u ~ v in comp if u != v and NOT (u ~ v in G)
    adj_comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                adj_comp[i].add(j)
                # adj_comp[j].add(i) # Symmetric, so add both or just one and use symmetric logic
                # Let's just build full sets for simplicity in Bron-Kerbosch
    
    # Function to check if alpha(G) >= 11
    # Equivalent to: does comp(G) have a clique of size 11?
    
    def has_clique_size_k(adj_graph, k):
        # Bron-Kerbosch with pivot
        # Returns True if a clique of size k is found
        
        def bronk(R, P, X, depth):
            if len(R) >= k:
                return True
            if not P and not X:
                return False
            
            # Pivot selection: choose u in P union X with max degree in P
            U = P.union(X)
            if not U:
                return False
            
            # Simple pivot: first element
            u = next(iter(U))
            # Better: max degree
            # For speed, just pick one
            candidates = P.difference(adj_graph[u])
            
            for v in list(candidates):
                if len(R) + len(P) < k: # Pruning
                    continue
                if bronk(R.union([v]), P.intersection(adj_graph[v]), X.intersection(adj_graph[v]), depth+1):
                    return True
                P.remove(v)
                X.add(v)
            return False
        
        return bronk(set(), set(range(len(adj_graph))), set(), 0)

    # Check 43-vertex graph
    if time.time() - start_time > deadline:
        # Fallback to seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    is_invalid_43 = has_clique_size_k(adj_comp, 11)
    
    if not is_invalid_43:
        # Valid 43-vertex graph
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # If invalid, we need to find a subgraph.
    # Strategy: Greedily remove vertices that are in large independent sets.
    # Or simply try to find a valid subset of size 39.
    # We can iterate through combinations of 4 vertices to remove.
    # There are C(43,4) ~ 123,410 combinations.
    # Checking each with Bron-Kerbosch might be slow.
    # Optimization: Only check if the removal breaks all known independent sets?
    # Simpler: Just try a few random removals or systematic ones.
    # Since 39 is the target, let's try to find a 39-vertex induced subgraph with alpha < 11.
    
    # Let's try removing 4 vertices.
    # To speed up, we can pre-calculate some properties or use a heuristic.
    # But for now, let's just try a few specific removals or random ones if time permits.
    # Actually, if the 43-vertex graph has alpha >= 11, it's likely that removing a few vertices fixes it.
    
    # Let's try to find a maximal independent set in G (clique in comp) and remove one vertex from it.
    # Repeat until alpha < 11.
    
    current_nodes = set(range(n))
    current_adj_comp = {i: adj_comp[i].intersection(current_nodes) for i in current_nodes}
    
    # Function to find one independent set of size 11 (clique in comp)
    def find_clique_size_k(adj_graph, k, nodes):
        # Similar to has_clique but returns the set
        def bronk(R, P, X):
            if len(R) >= k:
                return R
            if not P and not X:
                return None
            
            U = P.union(X)
            if not U:
                return None
            
            u = next(iter(U))
            candidates = P.difference(adj_graph[u])
            
            for v in list(candidates):
                if len(R) + len(P) < k:
                    continue
                res = bronk(R.union([v]), P.intersection(adj_graph[v]), X.intersection(adj_graph[v]))
                if res:
                    return res
                P.remove(v)
                X.add(v)
            return None
        
        return bronk(set(), set(nodes), set())

    attempts = 0
    while time.time() - start_time < deadline - 5:
        if len(current_nodes) < 11:
            break # Should not happen
        
        # Check if current graph has alpha >= 11
        # We can just try to find a clique of size 11
        clique = find_clique_size_k(current_adj_comp, 11, current_nodes)
        
        if clique is None:
            # Valid!
            # Extract edges from original adj restricted to current_nodes
            nodes_list = sorted(list(current_nodes))
            idx_map = {v: i for i, v in enumerate(nodes_list)}
            edges = []
            for i in range(len(nodes_list)):
                u = nodes_list[i]
                for v in adj[u]:
                    if v in idx_map and idx_map[v] > i:
                        edges.append([i, idx_map[v]])
            return {'n': len(nodes_list), 'edges': edges}
        
        # Found a clique (independent set in G). Remove one vertex.
        # Which one? The one that appears in the most cliques?
        # Just remove the first one for now.
        v_to_remove = next(iter(clique))
        current_nodes.remove(v_to_remove)
        # Update current_adj_comp
        for u in current_nodes:
            current_adj_comp[u].discard(v_to_remove)
        del current_adj_comp[v_to_remove]
        
        attempts += 1
        if attempts > 50:
            break # Safety

    # Fallback if loop fails or times out
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
