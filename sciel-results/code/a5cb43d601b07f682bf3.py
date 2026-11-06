"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _build_edge_index():
    n = P['n']
    edge_idx = {}
    for i, (u, v) in enumerate(P['edge_order']):
        edge_idx[(min(u,v), max(u,v))] = i + 1
    return edge_idx

_edge_idx = _build_edge_index()

def _find_hamilton_cycle_threshold(K, timeout=5.0):
    """
    Find a Hamilton cycle using only edges with index <= K.
    Uses randomized backtracking with pruning.
    """
    n = P['n']
    # Build adjacency list for edges <= K
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(P['edge_order']):
        if i + 1 > K:
            break
        adj[u].append(v)
        adj[v].append(u)
    
    # Pre-check: All vertices must have degree >= 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None
            
    start_time = time.time()
    rng = random.Random(K)
    
    for attempt in range(200): # 200 restarts
        if time.time() - start_time > timeout:
            break
            
        # Randomize neighbor order for this attempt
        adj_rand = [list(nei) for nei in adj]
        for i in range(n):
            rng.shuffle(adj_rand[i])
            
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(u, depth):
            if depth == n - 1:
                return 0 in adj_rand[u]
            
            # Pruning: Check for isolated vertices in the unvisited set
            # This is O(N^2) per step, but N=24 is small.
            # We check if any unvisited vertex has no valid next steps.
            # A vertex v is "stuck" if it has no neighbors in unvisited U {u}
            # (unless it's the last one and connects to 0).
            
            # Optimization: Only do this check every 2 steps or if depth is high?
            # For now, do it every step to maximize pruning.
            if depth > 0:
                unvisited = [i for i in range(n) if not visited[i]]
                for v in unvisited:
                    # Count neighbors that are either unvisited or the current node u
                    # (since we can come from u to v)
                    # Actually, if we are at u, and v is unvisited, v can be next.
                    # But if v is NOT next, v must be reachable later.
                    # If v has no neighbors in unvisited U {u}, it can only be visited if it is the immediate next step?
                    # No, if v has no neighbors in unvisited, it must be connected to u to be visited at all.
                    # If v is not connected to u, and has no unvisited neighbors, it's isolated.
                    deg = 0
                    for w in adj_rand[v]:
                        if not visited[w] or w == u:
                            deg += 1
                            
                    if deg == 0:
                        return False
                        
                    # If v is the last vertex to be visited, it must connect to 0.
                    if len(unvisited) == 1:
                        if v not in adj_rand[0]:
                            return False
                            
            for v in adj_rand[u]:
                if not visited[v]:
                    # Pruning: If v is the last vertex, it must connect to 0
                    if depth == n - 2:
                        if 0 not in adj_rand[v]:
                            continue
                            
                    visited[v] = True
                    path.append(v)
                    if dfs(v, depth + 1):
                        return True
                    path.pop()
                    visited[v] = False
            return False
            
        if dfs(0, 0):
            return path
            
    return None

def construct():
    n = P['n']
    max_edge_idx = 276
    
    # Binary search for the minimum K
    low = 24
    high = 276
    best_cycle = None
    best_K = max_edge_idx + 1
    
    # Use a shorter timeout for intermediate steps in binary search
    # to ensure we finish in 60s total.
    # 20 iterations * 5s = 100s, which is too long.
    # We should use a smaller timeout or fewer restarts.
    # Let's set timeout to 2.0s per call. 20 * 2 = 40s. Safe.
    
    for _ in range(20):
        mid = (low + high) // 2
        cycle = _find_hamilton_cycle_threshold(mid, timeout=2.0)
        if cycle:
            best_cycle = cycle
            best_K = mid
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        # Fallback
        rng = random.Random(614)
        cycle = list(range(n))
        rng.shuffle(cycle)
        return cycle
        
    return best_cycle
# EVOLVE-BLOCK-END
