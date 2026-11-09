import random
import time

def construct():
    n = 20
    deadline = time.time() + 110  # 110s safety margin
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(adj, u, v):
        # Check if adding edge (u, v) violates the condition.
        # A violation occurs if there exists a triple (a, b, c) such that
        # the new intersection of their neighbors has size >= 3.
        # The new neighbors are the old neighbors plus the new edge.
        # So u is added to N(v) and v is added to N(u).
        # This can only increase the common neighbor count for triples 
        # that include u or v? No.
        # Let's look at the condition: |N(a) & N(b) & N(c)| <= 2.
        # Adding (u,v) changes N(u) and N(v).
        # It affects any triple (a,b,c) where u or v is one of the vertices, 
        # OR where u or v is a common neighbor.
        
        # Case 1: The triple includes u.
        # Triples (u, x, y).
        # New N(u) = Old N(u) | {v}.
        # Common neighbors of (u, x, y) = (Old N(u) | {v}) & N(x) & N(y).
        # This is (Old N(u) & N(x) & N(y)) | ({v} & N(x) & N(y)).
        # The size increases by 1 if v is in N(x) & N(y) and v was NOT in Old N(u) & N(x) & N(y).
        # Since (u,v) is not an edge yet, v is not in Old N(u).
        # So if v is a neighbor of both x and y, then the common neighbor count for (u,x,y) increases by 1.
        # We need the new count to be <= 2.
        # So if the old count was 2, and v is a common neighbor of x and y, then new count is 3 -> Invalid.
        
        # Case 2: The triple includes v.
        # Symmetric to Case 1.
        
        # Case 3: The triple does not include u or v.
        # Let triple be (a, b, c) with a,b,c != u,v.
        # New N(a), N(b), N(c) are unchanged.
        # So the common neighbor count is unchanged.
        # Wait. The common neighbors are vertices w such that w is connected to a, b, c.
        # Adding (u,v) does not change the neighbors of a, b, c.
        # So the set of common neighbors of (a,b,c) is unchanged.
        # So no new violations are created for triples not containing u or v.
        
        # Therefore, we only need to check triples containing u or v.
        
        # Check triples (u, x, y) for all x, y != u, x != y.
        for x in range(n):
            if x == u: continue
            for y in range(x+1, n):
                if y == u: continue
                # Check if adding (u,v) makes (u,x,y) have >= 3 common neighbors.
                # Current common neighbors of (u,x,y): cn_old = adj[u] & adj[x] & adj[y]
                # New common neighbors: cn_new = cn_old | (1 << v) if v is neighbor of x and y.
                # Actually, v is added to N(u).
                # So v is a common neighbor of (u,x,y) iff v is in N(x) and v is in N(y).
                if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
                    # v becomes a common neighbor.
                    # Check if there are already 2 other common neighbors.
                    # cn_old = adj[u] & adj[x] & adj[y]
                    # Note: v is NOT in adj[u] yet.
                    # So cn_old does not contain v.
                    # If popcount(cn_old) >= 2, then adding v makes it >= 3.
                    if popcount(adj[u] & adj[x] & adj[y]) >= 2:
                        return False
        
        # Check triples (v, x, y)
        for x in range(n):
            if x == v: continue
            for y in range(x+1, n):
                if y == v: continue
                if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
                    if popcount(adj[v] & adj[x] & adj[y]) >= 2:
                        return False
                        
        return True

    best_edges = 0
    best_graph = None
    
    while time.time() < deadline:
        # Random restart
        # Start with a random graph or empty?
        # Starting empty allows us to add many edges.
        # Let's start with a random sparse graph to break symmetry.
        adj = [0] * n
        # Add some random edges initially?
        # Or just start empty and greedily add.
        
        # Greedy add edges in random order
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        current_edges = 0
        for u, v in edges_to_try:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = adj[:]
            
        # Optional: Local search improvement
        # Try to swap edges or add more if possible (already did greedy)
        # The greedy pass might get stuck.
        # We can do a second pass with a different order.
        # But random restarts are simpler.
        
    # Convert best_graph to list of edges
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    res = []
    for u in range(n):
        mask = best_graph[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                res.append([u, v])
            mask ^= lsb
    return res
