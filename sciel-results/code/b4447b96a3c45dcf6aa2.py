import time
import random
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = None
    best_count = 0

    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def count_edges(A):
        return int(np.sum(A)) // 2

    def is_valid(A):
        # Check all pairs
        # A^2 diagonal is degree, off-diagonal is common neighbors
        # We need off-diagonal elements of A @ A to be <= 2
        # But A is boolean, so A @ A counts common neighbors
        common_counts = A @ A
        # Set diagonal to 0 to ignore self-loops
        np.fill_diagonal(common_counts, 0)
        return np.max(common_counts) <= 2

    def try_greedy_fill(A):
        # Try to add all edges that can be added
        # We iterate until no more edges can be added
        added = True
        while added:
            added = False
            # Shuffle edges to break symmetry
            random.shuffle(all_edges)
            for u, v in all_edges:
                if A[u, v]:
                    continue
                # Check if can add
                # 1. Direct common neighbors
                if np.sum(A[u] & A[v]) > 2:
                    continue
                
                # 2. Neighbors of u
                N_u = np.where(A[u])[0]
                if len(N_u) > 0:
                    # Check if any w in N_u has > 1 common neighbor with v
                    # Common neighbors of w and v: sum(A[w] & A[v])
                    # Note: u is in N_w, but u is not in N_v yet.
                    # So current common(w, v) does not include u.
                    # After adding (u,v), u will be in N_v, so common(w,v) increases by 1.
                    # So we need current common(w,v) <= 1.
                    counts = np.sum(A[N_u] & A[v], axis=1)
                    if np.any(counts > 1):
                        continue

                # 3. Neighbors of v
                N_v = np.where(A[v])[0]
                if len(N_v) > 0:
                    counts = np.sum(A[u] & A[N_v], axis=1)
                    if np.any(counts > 1):
                        continue

                A[u, v] = True
                A[v, u] = True
                added = True
        return A

    def repair(A):
        # Remove edges causing violations
        # Find violations
        common_counts = A @ A
        np.fill_diagonal(common_counts, 0)
        
        violations_found = True
        while violations_found:
            violations_found = False
            # Find pairs with > 2 common neighbors
            # We need to remove an edge incident to one of the common neighbors
            # or one of the pair?
            # Removing (u, v) doesn't help if u, v are the pair with common neighbors.
            # We need to remove an edge (w, x) where w is a common neighbor of u, v?
            # If we remove (u, w), then w is no longer a common neighbor of u, v?
            # Yes.
            
            # Get list of violating pairs
            viol_pairs = np.argwhere(common_counts > 2)
            
            if len(viol_pairs) == 0:
                break
            
            # Pick a random violation
            u, v = viol_pairs[0]
            
            # Get common neighbors
            common = np.where(A[u] & A[v])[0]
            
            # Pick a random common neighbor w
            w = random.choice(common)
            
            # Remove edge (u, w) or (v, w)
            if random.random() < 0.5:
                A[u, w] = False
                A[w, u] = False
            else:
                A[v, w] = False
                A[w, v] = False
            
            # Recalculate common counts
            common_counts = A @ A
            np.fill_diagonal(common_counts, 0)
            violations_found = True

        return A

    def local_search(A):
        # Try to swap edges to increase count
        # This is complex, so we rely on restarts and greedy fill
        pass

    best_adj = np.zeros((n, n), dtype=bool)
    
    restarts = 0
    max_restarts = 200 # Adjust based on time
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        # Initialize with random graph
        # Density around 0.4-0.5
        p = random.uniform(0.3, 0.6)
        A = np.random.rand(n, n) < p
        A = (A | A.T) & (1 - np.eye(n, dtype=bool))
        
        # Repair
        A = repair(A)
        
        # Greedy Fill
        A = try_greedy_fill(A)
        
        curr_count = count_edges(A)
        
        if curr_count > best_count:
            best_count = curr_count
            best_adj = A.copy()
            
        # Check if we hit the theoretical max?
        # ex(20, K2,3) is likely 80 or 82.
        # If we find 80+, we can stop early if time allows or just continue to see if we can go higher.
        if best_count >= 85:
             break

    # Convert to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return edges
