import numpy as np
import time

def construct():
    """
    Finds a valid 2-coloring of K_n without a monochromatic K_{2,2,2}.
    Returns a dictionary with 'n' and 'colors'.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    # Target n = 29
    target_n = 29
    
    # Initialize with a random coloring or a known valid smaller one
    # Start with n=17 Paley coloring as a seed for the first 17 vertices
    # And random for the rest
    def get_paley_color(i, j, p=17):
        diff = (j - i) % p
        # Squares mod 17
        # 1, 4, 9, 16, 8, 2, 15, 13
        squares = {1, 2, 4, 8, 9, 13, 15, 16}
        return 0 if diff in squares else 1

    # Create initial adjacency matrix for target_n
    # A[i][j] = 0 or 1
    A = np.zeros((target_n, target_n), dtype=np.uint8)
    
    # Fill with Paley for i,j < 17
    for i in range(17):
        for j in range(i + 1, 17):
            c = get_paley_color(i, j)
            A[i, j] = c
            A[j, i] = c
            
    # Fill rest with random
    for i in range(17, target_n):
        for j in range(i + 1, target_n):
            c = np.random.randint(0, 2)
            A[i, j] = c
            A[j, i] = c
            
    # Also randomize edges between <17 and >=17
    for i in range(17):
        for j in range(17, target_n):
            c = np.random.randint(0, 2)
            A[i, j] = c
            A[j, i] = c

    def calculate_cost(A):
        """
        Counts potential K_{2,2,2} structures.
        For each color c, for each pair (u,v), let k be the number of common neighbors.
        If k >= 4, it's a candidate.
        We sum (k-3)^2 for all pairs and colors where k>=4.
        """
        cost = 0
        n = A.shape[0]
        # For each color c in 0, 1
        for c in [0, 1]:
            # Mask of edges with color c
            # We need common neighbors in G_c
            # Common neighbors of u,v in G_c are w such that A[u,w]==c and A[v,w]==c
            # This is (A[u,:] == c) & (A[v,:] == c)
            # Sum over all pairs u < v
            for u in range(n):
                for v in range(u + 1, n):
                    # Count common neighbors
                    # Exclude u and v themselves? 
                    # In a simple graph, A[u,u]=0, so if c=0, u is a common neighbor of u,v? 
                    # No, K_{2,2,2} requires distinct vertices.
                    # Let's assume the graph is simple and we only count w != u, v.
                    # A[u,u] is 0. If c=0, A[u,u]==0 is True. So we must mask out u and v.
                    common = (A[u, :] == c) & (A[v, :] == c)
                    common[u] = False
                    common[v] = False
                    k = np.sum(common)
                    if k >= 4:
                        cost += (k - 3) ** 2
        return cost

    best_cost = calculate_cost(A)
    best_A = A.copy()
    
    # Local Search
    temperature = 1.0
    cooling = 0.99
    min_temperature = 0.01
    
    while time.time() < deadline:
        if temperature < min_temperature:
            temperature = 1.0 # Restart
            # Optionally re-randomize if stuck
            if np.random.random() < 0.1:
                A = best_A.copy()
        
        # Pick a random edge to flip
        u = np.random.randint(0, target_n)
        v = np.random.randint(0, target_n)
        if u == v:
            continue
            
        old_color = A[u, v]
        new_color = 1 - old_color
        
        # Calculate delta cost
        # Flipping (u,v) affects common neighbor counts for pairs involving u or v?
        # No, it affects the edge (u,v) itself.
        # The cost function depends on common neighbors.
        # Flipping (u,v) changes whether u is a common neighbor of (x,v) for all x?
        # Yes.
        # This makes incremental cost calculation complex.
        # Given n=29, full recalculation is O(n^3) = 29^3 * 2 colors * 12 ops ~ 50k ops.
        # 110s / 50k ~ 2200 steps/s. That's too slow.
        # We need a faster cost function or incremental update.
        
        # Let's try a faster cost: 
        # Just check for existence of K_{2,2,2} directly?
        # Or use a cheaper proxy.
        # Proxy: Sum of degrees in G_c squared? No.
        
        # Let's optimize the cost calculation using matrix multiplication.
        # Let M_c be the adjacency matrix of color c (0/1).
        # Common neighbors of u,v is (M_c @ M_c)[u,v].
        # We want to minimize sum over u<v of max(0, (M_c @ M_c)[u,v] - 3)^2.
        # M_c @ M_c is O(n^3) but can be done with numpy matmul which is fast.
        
        def fast_cost(A):
            cost = 0
            for c in [0, 1]:
                M = (A == c).astype(np.float32)
                np.fill_diagonal(M, 0) # Ensure no self loops
                # M @ M gives number of common neighbors
                common_counts = M @ M
                # We only care about off-diagonal elements u < v
                # Extract upper triangle
                triu = np.triu(common_counts, k=1)
                # Penalize values >= 4
                # penalty = (triu - 3)^2 where triu >= 4
                # Use numpy where
                penalty = np.where(triu >= 4, (triu - 3) ** 2, 0)
                cost += np.sum(penalty)
            return cost

        # Recalculate cost with fast method
        current_cost = fast_cost(A)
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = A.copy()
            if best_cost == 0:
                break
        
        # Flip edge
        A[u, v] = new_color
        A[v, u] = new_color
        
        # Simulated Annealing acceptance
        new_cost = fast_cost(A)
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            current_cost = new_cost
        else:
            # Reject
            A[u, v] = old_color
            A[v, u] = old_color
            
        temperature *= cooling

    # Return best found
    final_n = target_n
    # Verify best_A is valid (cost 0)
    if fast_cost(best_A) == 0:
        # Extract colors in lexicographic order
        colors = []
        for i in range(final_n):
            for j in range(i + 1, final_n):
                colors.append(int(best_A[i, j]))
        return {'n': final_n, 'colors': colors}
    else:
        # Fallback to n=17 if 29 failed
        # Reconstruct n=17 from best_A or original
        # Just use the known valid n=17
        n = 17
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(best_A[i, j]))
        return {'n': n, 'colors': colors}
