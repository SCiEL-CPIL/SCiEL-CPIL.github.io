import time
import sys
from functools import lru_cache
sys.setrecursionlimit(10000)

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = 24
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Build adjacency with arrival indices
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(P['edge_order'], 1):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Precompute: for each vertex, list of (neighbor, arrival_index) sorted by arrival_index
    for u in range(n):
        adj[u].sort(key=lambda x: x[1])
    
    # Precompute bitmasks for quick neighbor checks
    # adj_mask[u] = bitmask of all neighbors of u
    # adj_arrival[u][v] = arrival index of edge (u,v), or large number if not present
    adj_mask = [0] * n
    adj_arrival = [[(1 << 30) for _ in range(n)] for __ in range(n)]
    for u in range(n):
        for v, idx in adj[u]:
            adj_mask[u] |= (1 << v)
            adj_arrival[u][v] = idx
            adj_arrival[v][u] = idx
    
    def get_neighbors_below(K, u):
        """Get list of neighbors of u with arrival index <= K, as bitmask."""
        mask = 0
        for v, idx in adj[u]:
            if idx <= K:
                mask |= (1 << v)
            else:
                break
        return mask
    
    def has_hamilton_cycle(K):
        """Check if there's a Hamilton cycle using only edges with arrival index <= K."""
        # Build adjacency masks for threshold K
        adj_K_mask = [0] * n
        adj_K_list = [[] for _ in range(n)]
        for u in range(n):
            mask = get_neighbors_below(K, u)
            adj_K_mask[u] = mask
            # Also build list for iteration
            for v in range(n):
                if (mask >> v) & 1:
                    adj_K_list[u].append(v)
        
        # Quick check: every vertex must have degree >= 2
        for u in range(n):
            if len(adj_K_list[u]) < 2:
                return None
        
        # DFS with aggressive pruning
        start = 0
        
        def dfs(current, visited, remaining, path):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                if (adj_K_mask[current] >> start) & 1:
                    return path[:]
                return None
            
            # Get unvisited vertices (excluding start)
            unvisited_mask = 0
            for v in range(1, n):
                if not (visited >> v) & 1:
                    unvisited_mask |= (1 << v)
            
            if unvisited_mask == 0:
                return None
            
            # Pruning 1: Each unvisited vertex must have at least 2 neighbors in (unvisited ∪ {current, start})
            # because it needs to be part of the path
            # Actually, in a Hamilton path from current through unvisited back to start:
            # - Internal unvisited vertices need 2 neighbors in the remaining structure
            # - Endpoints need 1
            # The endpoints are 'current' and 'start', so all unvisited are internal
            # Each unvisited vertex must have at least 2 neighbors in (unvisited ∪ {current, start})
            
            for v in range(1, n):
                if (unvisited_mask >> v) & 1:
                    # Count neighbors in (unvisited ∪ {current, start})
                    neighbor_mask = adj_K_mask[v] & (unvisited_mask | (1 << current) | (1 << start))
                    # Remove v itself
                    neighbor_mask &= ~(1 << v)
                    count = bin(neighbor_mask).count('1')
                    if count < 2:
                        return None
            
            # Pruning 2: Check that the subgraph induced by unvisited ∪ {current} is connected
            # (necessary for a Hamilton path to exist through them)
            if remaining > 1:
                # BFS from current through unvisited vertices
                visited_check = 1 << current
                stack = [current]
                while stack:
                    x = stack.pop()
                    nbs = adj_K_mask[x] & unvisited_mask
                    while nbs:
                        lsb = nbs & (-nbs)
                        v = lsb.bit_length() - 1
                        if not (visited_check >> v) & 1:
                            visited_check |= (1 << v)
                            stack.append(v)
                        nbs ^= lsb
                # All unvisited must be reachable
                if (visited_check & unvisited_mask) != unvisited_mask:
                    return None
            
            # Choose next vertex: neighbors of current that are unvisited
            candidates_mask = adj_K_mask[current] & unvisited_mask
            
            if candidates_mask == 0:
                return None
            
            # Get candidate list and sort by heuristic
            candidates = []
            cm = candidates_mask
            while cm:
                lsb = cm & (-cm)
                v = lsb.bit_length() - 1
                # Heuristic: prefer vertices with fewer options (minimum degree in remaining)
                remaining_nbs = adj_K_mask[v] & (unvisited_mask | (1 << start))
                remaining_nbs &= ~(1 << v)
                deg = bin(remaining_nbs).count('1')
                candidates.append((deg, v))
                cm ^= lsb
            
            candidates.sort()
            
            for _, nb in candidates:
                new_visited = visited | (1 << nb)
                path.append(nb)
                result = dfs(nb, new_visited, remaining - 1, path)
                path.pop()
                if result:
                    return result
            
            return None
        
        # Try different second vertices
        # Sort by some heuristic
        second_candidates = []
        sm = adj_K_mask[start]
        while sm:
            lsb = sm & (-sm)
            v = lsb.bit_length() - 1
            second_candidates.append(v)
            sm ^= lsb
        
        # Try in order of increasing degree (constrain early)
        second_candidates.sort(key=lambda v: bin(adj_K_mask[v] & ~(1 << start)).count('1'))
        
        for second in second_candidates:
            path = [start, second]
            visited = (1 << start) | (1 << second)
            result = dfs(second, visited, n - 2, path)
            if result:
                return result
        
        return None
    
    # Binary search on K
    lo, hi = 1, 276
    best_K = 276
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = has_hamilton_cycle(mid)
        if cycle:
            best_K = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is None:
        # Fallback: try a greedy approach with the full edge set
        return list(range(n))
    
    return best_cycle
