import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find a C4-free graph on 46 vertices with min degree >= 7
    n_target = 46
    min_deg_target = n_target - 39  # 7
    
    # Use a randomized construction
    # Start with an empty graph and add edges?
    # Or start with a random graph and remove C4s?
    
    # Better: Start with a known structure or random regular graph
    # Let's try a random 7-regular graph on 46 vertices and remove C4s.
    # If we can maintain min degree 7, we win.
    
    def has_c4(adj_matrix):
        # Check for C4 using matrix multiplication or direct check
        # adj_matrix is boolean numpy array
        # C4 exists if (A^2)[i,j] > 0 and A[i,j] == 0? No, that's a path of length 2.
        # C4 is a cycle of length 4.
        # A[i,j] * (A^2)[i,j] counts paths of length 2 between i and j which are adjacent? No.
        # Number of C4s is 1/8 * sum_{i,j} (A^2)_{ij} (A^2)_{ji} - ...
        # Simpler: For each pair of vertices (u,v) not adjacent, if they have 2 common neighbors, it's a C4.
        # Or for each edge (u,v), if they have a common neighbor, it's a triangle.
        # C4: u-v-w-x-u.
        # Common neighbors of u and w (non-adjacent) form a C4 if u-w is not an edge?
        # If u and w have 2 common neighbors v, x, and u-w is not an edge, then u-v-w-x-u is a C4.
        # If u-w is an edge, it's a diamond (K4 minus edge), which contains C4s? No, K4-e has no C4.
        # Wait, a C4 is an induced cycle or just a cycle? "Containing no C4" usually means no subgraph isomorphic to C4.
        # If u and w have 2 common neighbors, the subgraph induced by {u, v, w, x} contains a C4 u-v-w-x-u regardless of whether u-w is an edge?
        # If u-w is an edge, the edges are u-v, v-w, w-x, x-u, u-w.
        # The cycle u-v-w-x-u is present. So yes, 2 common neighbors implies a C4 subgraph.
        
        A = adj_matrix
        A2 = A @ A
        # For each pair i, j, if A2[i,j] >= 2, then there are at least 2 common neighbors.
        # This implies a C4 exists.
        # We just need to check if any A2[i,j] >= 2.
        return np.any(A2 >= 2)

    def get_min_degree(adj_matrix):
        return np.min(adj_matrix.sum(axis=1))

    def try_construct(n, target_deg):
        random.seed(42) # Reproducibility for debugging, maybe randomize later
        # Generate random graph
        # We want to end up with min degree >= target_deg
        # Start with a random graph with p=0.5
        adj = np.random.rand(n, n) < 0.5
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        np.fill_diagonal(adj, 0)
        
        # Remove C4s
        # Iteratively find a C4 and remove an edge
        # This might be slow.
        
        # Alternative: Start with empty graph, add edges greedily if no C4 created.
        # This ensures C4-free.
        # Then check min degree.
        # If min degree < target, try to add more edges.
        
        adj = np.zeros((n, n), dtype=bool)
        
        # List of all possible edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u,v) creates a C4
            # (u,v) creates a C4 if u and v have a common neighbor? No, that's a triangle.
            # (u,v) creates a C4 if there exists w such that u-w and v-w are edges? No.
            # (u,v) creates a C4 if there exist w, x such that u-w, w-x, x-v are edges?
            # Specifically, if u and v have a common neighbor w, adding u-v creates a triangle.
            # A C4 is formed if u and v have a common neighbor w, AND w has a neighbor x such that x is connected to u or v?
            # Let's use the common neighbor check.
            # Adding edge (u,v) creates a C4 if the number of common neighbors of u and v is >= 1?
            # No. If u and v have 1 common neighbor w, we have u-w-v. Adding u-v makes a triangle.
            # If u and v have 2 common neighbors w, x, we have u-w-v-x-u? No, w and x are connected to u and v.
            # Edges: u-w, w-v, u-x, x-v.
            # Adding u-v.
            # Cycle: u-w-v-x-u. This is a C4.
            # So if u and v have >= 2 common neighbors, adding u-v creates a C4.
            # What if they have 1 common neighbor w?
            # Is there another path of length 3 between u and v?
            # If there is a path u-a-b-v of length 3, and we add u-v, we get a C4 u-a-b-v-u.
            # So we must check if there is a path of length 3 between u and v.
            
            # Efficient check:
            # Common neighbors of u: N(u)
            # Neighbors of N(u): N2(u)
            # If v is in N2(u), there is a path of length 2. (Triangle if v in N(u))
            # If v is in N3(u), there is a path of length 3.
            
            # For n=46, we can just check all pairs w, x.
            # C4 exists if there are w, x such that u-w, w-x, x-v are edges.
            # We can precompute or just loop.
            
            creates_c4 = False
            # Check paths of length 3: u-w-x-v
            # w != v, x != u
            # adj[u,w] and adj[w,x] and adj[x,v]
            # To speed up, use matrix multiplication?
            # Or just check common neighbors of u and v?
            # If they have 2 common neighbors, C4.
            # If they have 1 common neighbor w, check if w has another neighbor x connected to v? No, x must be connected to u?
            # Path u-w-x-v.
            # w is common neighbor? No, w is neighbor of u. x is neighbor of w. v is neighbor of x.
            # So we need to check if there is a path of length 3.
            
            # For small n, simple check is fine.
            # But we do this for many edges.
            
            # Optimization:
            # Let A be the adjacency matrix.
            # Number of paths of length 3 between u and v is (A^3)[u,v].
            # If (A^3)[u,v] > 0, adding u-v creates a C4.
            # Also if (A^2)[u,v] >= 2, adding u-v creates a C4 (via 2 common neighbors).
            # Actually, if (A^2)[u,v] >= 1, it's a triangle or part of C4?
            # If (A^2)[u,v] = 1, say common neighbor w.
            # If there is another path of length 3, we have C4.
            
            # So condition: (A^3)[u,v] > 0 OR (A^2)[u,v] >= 2.
            
            # We can maintain A2 and A3?
            # Updating A2 and A3 when adding an edge is O(n).
            
            # Let's just do the O(n^2) check for now, n=46 is small.
            # Check if any w, x exist.
            # w in N(u), x in N(w), v in N(x).
            
            # Faster:
            # For each w in N(u):
            #   For each x in N(w):
            #     if adj[x,v]: creates_c4 = True
            
            if creates_c4:
                continue
            
            # Also check if adding edge creates C4 via 2 common neighbors?
            # If u and v have 2 common neighbors, (A^2)[u,v] >= 2.
            # This is covered by the path of length 3?
            # If u-w-v and u-x-v.
            # Path u-w-v-x-u? No, x-v is edge, v-w is edge, w-u is edge, u-x is edge.
            # Cycle u-w-v-x-u.
            # This is a path of length 3 from u to v? u-w-v-x. No, x is connected to v.
            # The path is u-w-v-x. Length 3.
            # So yes, (A^3)[u,v] > 0 covers it.
            # Wait, if u-w-v and u-x-v.
            # Path u-w-v-x.
            # Edges: u-w, w-v, v-x.
            # Is v-x an edge? Yes, x is common neighbor, so v-x is edge.
            # So u-w-v-x is a path of length 3.
            # So checking (A^3)[u,v] > 0 is sufficient.
            
            # But wait, what if the path is u-w-x-v?
            # Then w and x are not necessarily common neighbors.
            # The condition for C4 is existence of path of length 3.
            
            # So:
            # if (A^3)[u,v] > 0: continue
            
            # Let's implement (A^3)[u,v] check efficiently.
            # We can just loop w in N(u), x in N(w) and check adj[x,v].
            
            flag = False
            for w in range(n):
                if adj[u, w]:
                    for x in range(n):
                        if adj[w, x] and adj[x, v]:
                            flag = True
                            break
                    if flag:
                        break
            
            if flag:
                continue
                
            adj[u, v] = True
            adj[v, u] = True
            
        return adj

    # Try to construct
    adj = try_construct(n_target, min_deg_target)
    min_deg = get_min_degree(adj)
    
    if min_deg >= min_deg_target:
        # Verify no C4
        A = adj.astype(int)
        A2 = A @ A
        if not np.any(A2 >= 2):
             # It is C4-free
             edges = []
             for i in range(n_target):
                 for j in range(i+1, n_target):
                     if adj[i, j]:
                         edges.append([i, j])
             return {'n': n_target, 'edges': edges}
             
    # If failed, try n=45, 44, etc.
    # Or just return a valid smaller graph.
    # For safety, return n=39 empty graph.
    return {'n': 39, 'edges': []}
