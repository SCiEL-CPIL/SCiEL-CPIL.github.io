import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    def check_c4(adj, u, v):
        """
        Check if adding edge (u, v) to graph with adjacency bitmasks 'adj'
        creates a C4.
        adj is a list of integers where bit i is set if edge (i, j) exists.
        """
        # We need to check if there is a path u-a-b-v
        # a in N(u), b in N(v), a~b
        # For each a in N(u), check if N(v) intersects N(a)
        # Note: u and v are not yet connected.
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Iterate through neighbors of u
        # We can iterate through set bits in mask_u
        a = 0
        while mask_u:
            if mask_u & 1:
                # a is a neighbor of u
                # Check if any neighbor of v is connected to a
                # N(v) is mask_v
                # N(a) is adj[a]
                if mask_v & adj[a]:
                    return True
            mask_u >>= 1
            a += 1
        return False

    def try_construct(n, min_deg_req, max_restarts=500):
        """
        Try to construct a C4-free graph on n vertices with min degree >= min_deg_req.
        Returns list of edges if successful, None otherwise.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            deg = [0] * n
            
            # We need to ensure min degree >= min_deg_req
            # Greedy approach:
            # Maintain a list of vertices that still need edges.
            # Repeatedly pick an edge that helps the most constrained vertex.
            
            # To speed up, we can just try to add all possible safe edges in a random order,
            # but bias towards low degree vertices.
            
            # List of all possible edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            # We will process edges in a priority queue manner or just iterate?
            # Iterating once might not be enough if we skip edges that are safe but don't help "now".
            # But any safe edge is good.
            # However, we must ensure we don't "use up" connectivity for a vertex
            # that might need it later.
            
            # Better strategy:
            # While min(deg) < min_deg_req:
            #   Find u with min deg.
            #   Find safe v to connect to u.
            #   If none, fail.
            
            success = True
            while min(deg) < min_deg_req:
                if time.time() > deadline:
                    return None
                
                # Find vertex with minimum degree
                u = min(range(n), key=lambda x: deg[x])
                
                if deg[u] >= min_deg_req:
                    break
                
                # Find candidates for v
                candidates = []
                # We want v such that (u,v) is safe.
                # To be efficient, we can check all v.
                for v in range(n):
                    if v == u:
                        continue
                    if (adj[u] >> v) & 1:
                        continue # already connected
                    
                    if check_c4(adj, u, v):
                        continue
                    
                    candidates.append(v)
                
                if not candidates:
                    success = False
                    break
                
                # Pick v with lowest degree to balance
                v = min(candidates, key=lambda x: deg[x])
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
            
            if success and min(deg) >= min_deg_req:
                # Verify C4-free property
                has_c4 = False
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            # Check if common neighbors > 1
                            common = adj[i] & adj[j]
                            # Popcount
                            if common & (common - 1): # if more than 1 bit set
                                has_c4 = True
                                break
                    if has_c4:
                        break
                
                if not has_c4:
                    # Convert to edge list
                    edges_list = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj[i] >> j) & 1:
                                edges_list.append([i, j])
                    return edges_list
        
        return None

    # Try n=47, 46, 45, 44...
    for n in [47, 46, 45, 44, 43]:
        if time.time() > deadline:
            break
            
        min_deg_req = n - 39
        
        edges = try_construct(n, min_deg_req)
        
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback: return a small valid graph
    # n=39, min_deg=0. Empty graph works.
    return {'n': 39, 'edges': []}
