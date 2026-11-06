import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    best_edges = 0
    best_graph = None
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6_path(u, v, adj, exclude_edge=None):
        """
        Checks if there is a path of length 4 between u and v in the graph defined by adj.
        If exclude_edge is (u, v), we ignore that edge in the adjacency lists for u and v.
        Path: u-a-b-c-v
        """
        # Get neighbors of u and v, excluding each other if exclude_edge is set
        nu = set(adj[u])
        nv = set(adj[v])
        
        if exclude_edge:
            nu.discard(v)
            nv.discard(u)
        
        if not nu or not nv:
            return False
            
        for a in nu:
            for c in nv:
                if a == c:
                    continue
                # Check intersection of N(a) and N(c)
                # We need b in N(a) and b in N(c)
                # b must be distinct from u, v, a, c
                # b cannot be u or v because:
                # If b==u, path u-a-u-c-v (not simple, a-u is edge, u-c is edge? No, u is start)
                # Actually, b is an intermediate vertex.
                # The vertices in the path are u, a, b, c, v.
                # They must be distinct.
                # So b != u, b != v, b != a, b != c.
                
                # Optimization: Pre-compute intersection?
                # For small n, just intersect.
                inter = adj[a] & adj[c]
                
                # Filter out u, v, a, c
                # Note: adj[a] contains u (since a in N(u)).
                # adj[c] contains v (since c in N(v)).
                # So u is likely in inter if u in N(c).
                # v is likely in inter if v in N(a).
                
                # We just need to see if there is ANY b in inter \ {u, v, a, c}
                # Since n=40, we can just iterate or filter.
                
                # Fast check:
                # If len(inter) > 0, we need to check if any element is valid.
                # Instead of creating a new set, we can just check:
                for b in inter:
                    if b != u and b != v and b != a and b != c:
                        return True
        return False

    def is_valid_addition(u, v, adj):
        """
        Checks if adding edge (u, v) creates a C6.
        This is equivalent to checking if there is a path of length 4 between u and v in the current graph.
        """
        return has_c6_path(u, v, adj, exclude_edge=None)

    def find_c6_edge(adj):
        """
        Finds an edge (u, v) such that removing it is necessary to break a C6?
        Actually, finds an edge (u, v) that is part of a C6.
        We check if there is a path of length 4 between u and v in G - (u, v).
        """
        for u, v in all_edges:
            if v in adj[u]:
                if has_c6_path(u, v, adj, exclude_edge=(u, v)):
                    return (u, v)
        return None

    for attempt in range(50): # Limit attempts
        if time.time() > deadline:
            break
            
        # 1. Initialize with random graph
        p = 0.5
        adj = [set() for _ in range(n)]
        edges = []
        
        # Randomly add edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                
        # 2. Remove edges that form C6s
        # We need to ensure the graph is C6-free.
        # Loop until no C6 is found.
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Find a C6
            # To be faster, we can pick a random edge and check?
            # Or iterate all edges.
            # With n=40, E~400. Checking each is ~40*40*small. 1600 ops. 400*1600 = 640k ops.
            # This is fast.
            
            # We need to find ANY edge that is part of a C6.
            # If we find one, remove it and restart the loop (or continue).
            # Removing one edge might break multiple C6s.
            
            # Optimization: Instead of scanning all edges every time,
            # we can just scan. If we find one, remove it.
            
            for u, v in shuffled_edges: # Use shuffled order for randomness
                if v in adj[u]:
                    if has_c6_path(u, v, adj, exclude_edge=(u, v)):
                        # Remove edge
                        adj[u].discard(v)
                        adj[v].discard(u)
                        edges.remove((u, v))
                        changed = True
                        break # Restart scan after removal? Or continue?
                        # Restarting is safer to ensure we don't miss overlapping C6s?
                        # Actually, just breaking and looping again is fine.
            
        # 3. Add edges greedily
        # Try to add all missing edges that don't create a C6
        # Order: Random or Degree-based? Random is fine.
        # We need to check all missing edges.
        
        # Create list of missing edges
        missing = [e for e in all_edges if e[1] not in adj[e[0]]]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if not is_valid_addition(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                
        # 4. Local Search / Polish
        # Try to add more edges? We just did.
        # Try to flip?
        # Given time, let's just record the best.
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = [[u, v] for u, v in edges]
            
    return best_graph if best_graph else []
