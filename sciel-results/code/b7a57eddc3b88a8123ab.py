"""Construct a C4-free graph with high minimum degree to certify R(C4,K1,39) > n."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    
    # Try to find a C4-free graph on n vertices with min degree >= n-39
    # Strategy: start with a dense graph (almost complete), then remove edges to break C4s
    # while maintaining high minimum degree
    
    def find_c4(adj_matrix):
        """Find a C4 in the graph. Returns (v1,v2,v3,v4) or None."""
        n = len(adj_matrix)
        # For each pair of vertices, check if they have 2 common neighbors
        # which would form a C4
        for v1 in range(n):
            for v2 in range(v1+1, n):
                if not adj_matrix[v1][v2]:
                    continue
                # Find common neighbors
                common = []
                for v3 in range(n):
                    if v3 != v1 and v3 != v2 and adj_matrix[v1][v3] and adj_matrix[v2][v3]:
                        common.append(v3)
                if len(common) >= 2:
                    v3, v4 = common[0], common[1]
                    # Verify it's a proper C4
                    if adj_matrix[v3][v4]:
                        # This is a K4, not just C4, but still contains C4
                        pass
                    return (v1, v2, v3, v4)
        return None
    
    def is_c4_free(adj_matrix):
        """Check if graph is C4-free."""
        n = len(adj_matrix)
        for v1 in range(n):
            for v2 in range(v1+1, n):
                if not adj_matrix[v1][v2]:
                    continue
                common_count = 0
                for v3 in range(n):
                    if v3 != v1 and v3 != v2 and adj_matrix[v1][v3] and adj_matrix[v2][v3]:
                        common_count += 1
                        if common_count >= 2:
                            return False
        return True
    
    def min_degree(adj_matrix):
        n = len(adj_matrix)
        min_deg = n
        for i in range(n):
            deg = sum(adj_matrix[i])
            min_deg = min(min_deg, deg)
        return min_deg
    
    def edges_from_adj(adj_matrix):
        n = len(adj_matrix)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i][j]:
                    edges.append([i, j])
        return edges
    
    def try_construct(n, max_iterations=50000):
        """Try to construct a C4-free graph on n vertices with min degree >= n-39."""
        if n <= 39:
            # Empty graph works
            return {'n': n, 'edges': []}
        
        required_min_deg = n - 39
        
        # Start with complete graph
        adj = np.ones((n, n), dtype=np.uint8)
        np.fill_diagonal(adj, 0)
        
        # Randomize initial edge removals to break symmetry
        # Remove some random edges first
        num_initial_removals = max(1, int(n * n * 0.05))
        for _ in range(num_initial_removals):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                adj[u, v] = 0
                adj[v, u] = 0
        
        # Iteratively remove edges to break C4s
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Check if we already satisfy min degree
            cur_min_deg = min_degree(adj)
            if cur_min_deg < required_min_deg:
                # We've removed too many edges, this path failed
                return None
            
            # Find a C4
            c4 = find_c4(adj)
            if c4 is None:
                # Success!
                return {'n': n, 'edges': edges_from_adj(adj)}
            
            v1, v2, v3, v4 = c4
            # Remove one edge from the C4. Choose the edge whose removal
            # is least likely to violate min degree
            # Options: (v1,v2), (v2,v3), (v3,v4), (v4,v1)
            # Also (v1,v3) and (v2,v4) if they exist, but let's stick to cycle edges
            
            candidates = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
            
            # Filter to only edges that exist
            valid_candidates = [(u, v) for u, v in candidates if adj[u, v]]
            
            if not valid_candidates:
                # Shouldn't happen, but break
                return None
            
            # Choose the edge whose endpoints have the highest degrees
            # (least critical for min degree)
            def edge_score(u, v):
                return (sum(adj[u]) + sum(adj[v]))
            
            valid_candidates.sort(key=edge_score, reverse=True)
            
            # Remove the first candidate
            u, v = valid_candidates[0]
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Check if min degree is still okay
            if min_degree(adj) < required_min_deg:
                # Restore and try another edge
                adj[u, v] = 1
                adj[v, u] = 1
                if len(valid_candidates) > 1:
                    u2, v2 = valid_candidates[1]
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    if min_degree(adj) < required_min_deg:
                        adj[u2, v2] = 1
                        adj[v2, u2] = 1
                        # All options fail, this path is dead
                        return None
                else:
                    return None
        
        return None
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple attempts with different random seeds
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            random.seed(attempt * 12345 + n)
            np.random.seed(attempt * 67890 + n)
            
            result = try_construct(n, max_iterations=30000)
            if result is not None:
                best_result = result
                break  # Found a valid graph for this n, try larger n next
        
        if best_result['n'] >= n:
            # We found a graph for this n, continue to try larger
            continue
    
    # If we didn't find anything better than 39, return 39
    if best_result['n'] < 39:
        best_result = {'n': 39, 'edges': []}
    
    return best_result
