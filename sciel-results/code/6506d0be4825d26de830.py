"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_list = P['edge_order']
total_edges = len(edge_list)

# Precompute adjacency lists for the full graph? No, we need dynamic subgraphs.
# But we can precompute the index of each edge to quickly check availability.
# Actually, for the "prefix" check, we just need to know if an edge is in the first K.
# We can represent the allowed edges as a set of frozensets or a boolean matrix.

# To speed up the search, we will build the adjacency structure incrementally or check on the fly.
# Given n=24, a boolean matrix (24x24) is very fast to check.

def exists_hamilton_cycle(k, deadline):
    """
    Checks if there is a Hamiltonian cycle using only the first k edges.
    Returns the cycle (list of vertices) if found, else None.
    """
    # Build adjacency matrix for the first k edges
    # adj[u][v] = True if edge (u,v) is in the first k
    adj = [[False] * n for _ in range(n)]
    
    # We iterate through the first k edges and mark them
    for i in range(k):
        u, v = edge_list[i]
        adj[u][v] = True
        adj[v][u] = True
        
    # Check if graph is connected enough? 
    # A necessary condition for a Hamiltonian cycle is that the graph is connected.
    # But for small k, it might be disconnected.
    # Let's do a quick connectivity check (BFS/DFS) on the induced subgraph.
    # If not connected, return None immediately.
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u][v] and not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    if count < n:
        return None

    # Backtracking search
    # We fix vertex 0 as the start to reduce symmetry.
    # path[0] = 0
    path = [0]
    on_path = [False] * n
    on_path[0] = True
    
    # Warnsdorff's rule: order neighbors by number of available neighbors (degree in remaining graph)
    # We need a fast way to get neighbors.
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        if len(path) == n:
            # Check if last vertex connects to first (0)
            if adj[path[-1]][0]:
                return True
            return False
        
        u = path[-1]
        
        # Get available neighbors
        neighbors = []
        for v in range(n):
            if not on_path[v] and adj[u][v]:
                # Calculate heuristic: number of neighbors of v that are also unvisited and connected to v
                # This is the "forward degree"
                deg = 0
                for w in range(n):
                    if not on_path[w] and adj[v][w]:
                        deg += 1
                neighbors.append((deg, v))
        
        if not neighbors:
            return False
            
        # Sort by degree (Warnsdorff)
        neighbors.sort()
        
        for deg, v in neighbors:
            on_path[v] = True
            path.append(v)
            if backtrack():
                return True
            path.pop()
            on_path[v] = False
            
        return False

    if backtrack():
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # Binary search for the minimum K
    # Lower bound: n (a cycle needs at least n edges)
    # Upper bound: total_edges
    lo = n
    hi = total_edges
    best_cycle = None
    best_k = hi
    
    # We can also try a linear scan from the bottom if we expect the answer to be small,
    # but binary search is safer for "existence" checks if the property is monotonic.
    # Property: If a HC exists in first K edges, it exists in first K+1 edges.
    # So we can binary search for the smallest K.
    
    # However, finding the *cycle* for a specific K might be hard.
    # If we find a cycle for K, we know the score is <= K.
    # We want the minimum K.
    
    # Optimization: Start with a random or greedy heuristic to get an initial upper bound?
    # The identity permutation [0..23] has a specific score. Let's calculate it to have a fallback.
    # Identity cycle: 0-1-2-...-23-0
    # Edges: (0,1), (1,2), ..., (22,23), (23,0)
    # Find their indices in edge_order.
    
    def score_cycle(cycle):
        # cycle is a list of vertices, implicit closing edge to cycle[0]
        max_idx = 0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            # Find index of edge (u,v)
            # This is slow if done naively. Precompute edge indices.
            pass
        return max_idx

    # Precompute edge indices for faster scoring if needed, but we are searching for existence.
    
    # Let's just do the binary search.
    # To speed up, we can check mid points.
    # If exists(mid) is True, hi = mid, best_cycle = cycle_found
    # If exists(mid) is False, lo = mid + 1
    
    # Note: exists_hamilton_cycle returns the cycle if found.
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        
        # Try to find a cycle in the first mid edges
        cycle = exists_hamilton_cycle(mid, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback: return a valid cycle, even if not optimal.
        # The identity permutation is a valid cycle.
        return list(range(n))
        
    return best_cycle

# EVOLVE-BLOCK-END
