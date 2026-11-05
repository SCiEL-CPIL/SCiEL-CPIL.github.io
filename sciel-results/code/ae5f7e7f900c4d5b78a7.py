import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin

    # Fallback: n=11 with a known valid coloring
    # For n=11, we can use a simple construction
    n_fallback = 11
    best_n = n_fallback
    # Simple valid coloring for n=11: use modular arithmetic
    best_colors = []
    for i in range(n_fallback):
        for j in range(i + 1, n_fallback):
            # Use a construction that avoids C4
            # For small n, random + repair should work, but let's have a deterministic fallback
            best_colors.append((i * 7 + j * 13) % 5)
    
    def is_valid(colors, n):
        """Check if coloring has no monochromatic C4."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        # Build adjacency sets for each color
        adj = [set() for _ in range(n)]
        color_adj = [set() for _ in range(n)]  # not quite right
        # Better: for each color, build adjacency list
        # Then check for C4 in each color class
        
        # For each color c, for each pair of vertices (u, v), check if they have >= 2 common neighbors in color c
        # A C4 exists if there are two vertices with at least 2 common neighbors in the same color
        
        # Build for each color: list of edges
        for c in range(5):
            # Build adjacency for color c
            adj_c = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                    if colors[idx] == c:
                        adj_c[i].add(j)
                        adj_c[j].add(i)
            
            # Check for C4: for each pair (u, v), count common neighbors
            # If any pair has >= 2 common neighbors, there's a C4
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_c[u] & adj_c[v]
                    if len(common) >= 2:
                        return False
        return True
    
    # Verify fallback
    if not is_valid(best_colors, n_fallback):
        # Try random search for n=11 as fallback
        for attempt in range(100):
            colors = [random.randint(0, 4) for _ in range(n_fallback * (n_fallback - 1) // 2)]
            if is_valid(colors, n_fallback):
                best_colors = colors
                break
    
    def solve_n_local_search(n, time_budget):
        """Local search with incremental C4 counting."""
        num_edges = n * (n - 1) // 2
        if num_edges == 0:
            return [0] * 0
        
        # Precompute edge endpoints
        endpoints = []
        for i in range(n):
            for j in range(i + 1, n):
                endpoints.append((i, j))
        
        # Edge index lookup: for edge (u, v) with u < v, find its index
        # index = sum_{k=0}^{u-1} (n-1-k) + (v-u-1)
        # = u*(n-1) - u*(u+1)//2 + v - u - 1
        # = u*n - u - u*(u+1)//2 + v - 1
        # Let me compute this differently
        # Number of edges from vertex k to vertices > k is n-1-k
        # For vertex i, edges to i+1, i+2, ..., n-1: that's n-1-i edges
        # Edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Wait: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - sum_{k=0}^{i-1} k = i*(n-1) - i*(i-1)//2
        # Then within row i, edge to j is at position (j - i - 1)
        # So index = i*(n-1) - i*(i-1)//2 + (j - i - 1)
        
        def edge_idx(u, v):
            if u > v:
                u, v = v, u
            return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
        
        # For each color, maintain:
        # - common_neighbors[c][u][v] = number of common neighbors of u and v in color c
        # This is expensive to store for large n, but for n <= 15 it's fine
        
        # Actually, let's use a different approach:
        # For each color c, maintain a matrix M_c where M_c[u,v] = number of common neighbors
        # A C4 in color c exists if M_c[u,v] >= 2 for some u != v
        # Cost = sum over all c, all u<v of C(M_c[u,v], 2)
        
        # For efficiency, let's track for each color and each pair (u,v) the count
        # But updating when we change an edge color is O(n) per color change
        
        # Let me use numpy for the common neighbor counts
        # CN[c] is an n x n matrix (symmetric) of common neighbor counts for color c
        
        def random_init():
            return [random.randint(0, 4) for _ in range(num_edges)]
        
        def compute_cost(colors):
            """Compute total C4 count across all colors."""
            total = 0
            for c in range(5):
                # Build adjacency for color c
                # For each vertex k, its neighbors in color c
                # Then for each pair of neighbors, increment common count
                cn = np.zeros((n, n), dtype=np.int16)
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        idx = edge_idx(i, j)
                        if colors[idx] == c:
                            adj[i].add(j)
                            adj[j].add(i)
                for k in range(n):
                    nbrs = list(adj[k])
                    for a in range(len(nbrs)):
                        for b in range(a + 1, len(nbrs)):
                            u, v = nbrs[a], nbrs[b]
                            cn[u, v] += 1
                            cn[v, u] += 1
                # Count C4s: sum of C(cn[u,v], 2) for u < v
                for u in range(n):
                    for v in range(u + 1, n):
                        cnt = cn[u, v]
                        if cnt >= 2:
                            total += cnt * (cnt - 1) // 2
            return total
        
        def compute_cost_fast(colors):
            """Faster cost computation using matrix multiplication."""
            total = 0
            for c in range(5):
                A = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        idx = edge_idx(i, j)
                        if colors[idx] == c:
                            A[i, j] = 1
                            A[j, i] = 1
                # M = A @ A gives common neighbor counts
                M = A @ A
                # For u != v, M[u,v] is the number of common neighbors
                # C4 count = sum_{u < v} C(M[u,v], 2) for u != v
                # But we need to exclude u == v cases (diagonal)
                # Diagonal entries M[u,u] = degree of u, which we don't count
                for u in range(n):
                    for v in range(u + 1, n):
                        cnt = M[u, v]
                        if cnt >= 2:
                            total += cnt * (cnt - 1) // 2
            return total
        
        # Local search with simulated annealing
        best_solution = None
        best_cost = float('inf')
        
        for restart in range(20):
            if time.time() > start_time + time_budget:
                break
            
            colors = random_init()
            current_cost = compute_cost_fast(colors)
            
            if current_cost == 0:
                return colors
            
            T = 5.0
            T_min = 0.1
            cooling = 0.9995
            
            accepted = 0
            for step in range(100000):
                if time.time() > start_time + time_budget:
                    break
                
                if current_cost == 0:
                    return colors
                
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                colors[e_idx] = new_c
                new_cost = compute_cost_fast(colors)
                
                delta = new_cost - current_cost
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    accepted += 1
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solution = colors.copy()
                else:
                    colors[e_idx] = old_c
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if best_solution is not None and best_cost == 0:
                return best_solution
        
        if best_solution is not None and best_cost == 0:
            return best_solution
        return None
    
    # Try to find solutions for increasing n
    for n in range(12, 20):
        if time.time() > deadline - 15:
            break
        
        # Allocate time based on n
        # Give more time to smaller n (easier)
        time_budget = min(8, deadline - time.time() - 5)
        if time_budget < 2:
            break
        
        result = solve_n_local_search(n, time_budget)
        if result is not None:
            # Verify
            if is_valid(result, n):
                best_n = n
                best_colors = result
            else:
                # Something went wrong, don't use this
                pass
    
    return {'n': best_n, 'colors': best_colors}
