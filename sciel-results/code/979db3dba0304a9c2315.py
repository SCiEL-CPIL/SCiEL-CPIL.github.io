import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_n = 39
    best_edges = []
    
    for target_n in [47, 46, 45, 44, 43]:
        if time.time() > deadline - 10:
            break
            
        success = False
        for attempt in range(1000):
            if time.time() > deadline - 5:
                break
                
            n = target_n
            min_deg_req = n - 39
            
            # Adjacency matrix
            A = np.zeros((n, n), dtype=np.int8)
            
            # List of edges to try
            # Randomize order to avoid bias
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
            np.random.shuffle(edges_to_try)
            
            # Greedy addition
            for u, v in edges_to_try:
                if time.time() > deadline - 2:
                    break
                    
                # Check if adding (u,v) creates a C4
                # C4 exists if there is an edge between N(u) and N(v)
                # N(u) are indices where A[u] == 1
                # N(v) are indices where A[v] == 1
                # We need to check if any A[w, x] == 1 for w in N(u), x in N(v)
                
                # Optimization: If degree is low, this is fast.
                # Using numpy slicing
                # Get neighbors
                # Note: A is symmetric.
                
                # Check intersection of neighborhoods? No, check edges between them.
                # If N(u) or N(v) is empty, no C4.
                if np.sum(A[u]) == 0 or np.sum(A[v]) == 0:
                    # Safe to add
                    A[u, v] = 1
                    A[v, u] = 1
                    continue
                
                # Check for edges between N(u) and N(v)
                # Exclude u and v from each other's neighbor lists if they were connected? 
                # They are not connected yet.
                # So N(u) and N(v) do not contain each other.
                
                # Create boolean masks
                # This might be slow if done in pure python loops, but numpy is C.
                # Let's try a vectorized check.
                # Submatrix A[N(u), N(v)]
                
                # Get indices
                # This creates views/copies.
                # Alternative: 
                # common = A[u] & A[v] -> this finds common neighbors (triangles).
                # We need to find if there is a path of length 2? No.
                # We need to find if there is an edge between N(u) and N(v).
                
                # Let's use the fact that for small n, we can just check.
                # But let's try to be smart.
                # If we add (u,v), a C4 is u-w-x-v-u.
                # This requires w in N(u), x in N(v), and (w,x) in E.
                
                # We can check:
                # For each w in N(u), check if N(w) intersects N(v).
                # This is sum over w in N(u) of (A[w] & A[v]).sum() > 0
                
                # Let's just do the numpy submatrix check, it's likely fast enough for n=47.
                # But extracting indices might be slow.
                
                # Alternative:
                # Count paths of length 2 between u and v?
                # Number of common neighbors is (A[u] * A[v]).sum() -> Triangles.
                # Number of C4s involving (u,v) is related to number of length-3 paths?
                
                # Let's just use a simple loop over neighbors of u.
                # deg(u) is small.
                neigh_u = np.nonzero(A[u])[0]
                if len(neigh_u) == 0:
                    A[u, v] = 1
                    A[v, u] = 1
                    continue
                
                # For each w in neigh_u, check if any neighbor of w is in neigh_v
                # neigh_v = np.nonzero(A[v])[0]
                # We can check if A[neigh_u, :][:, neigh_v] has any 1.
                
                # This creates a submatrix.
                # Let's try it.
                try:
                    # Get neighbors of v
                    neigh_v = np.nonzero(A[v])[0]
                    if len(neigh_v) == 0:
                        A[u, v] = 1
                        A[v, u] = 1
                        continue
                    
                    # Check submatrix
                    # A[neigh_u][:, neigh_v]
                    if np.any(A[np.ix_(neigh_u, neigh_v)]):
                        # C4 created, skip
                        continue
                    else:
                        A[u, v] = 1
                        A[v, u] = 1
                except Exception:
                    continue
            
            # Check min degree
            degrees = np.sum(A, axis=1)
            min_deg = np.min(degrees)
            
            if min_deg >= min_deg_req:
                # Valid graph found
                success = True
                best_n = n
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i, j]:
                            edges.append([i, j])
                best_edges = edges
                break # Try next n? No, we want max n. If we found 47, we are done.
        
        if success:
            break # Found max n, stop searching lower n
    
    if best_n == 39:
        return {'n': 39, 'edges': []}
    else:
        return {'n': best_n, 'edges': best_edges}
