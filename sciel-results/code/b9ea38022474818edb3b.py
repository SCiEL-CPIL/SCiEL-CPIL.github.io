import time
import random
import sys

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge index lookup: edge_index[u][v] = one-based index
edge_index = [[0] * N for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_index[u][v] = i
    edge_index[v][u] = i

# Build adjacency lists sorted by edge index
adj = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    adj[u].append((v, i))
    adj[v].append((u, i))
for v in range(N):
    adj[v].sort(key=lambda x: x[1])

def is_connected(K):
    """Check if graph with edges having index <= K is connected."""
    # Build adjacency for edges <= K
    visited = [False] * N
    queue = []
    # Find first vertex with an edge
    start = -1
    for i in range(N):
        has_edge = False
        for nb, idx in adj[i]:
            if idx <= K:
                has_edge = True
                break
        if has_edge:
            start = i
            break
    if start == -1:
        return N == 1
    
    queue.append(start)
    visited[start] = True
    count = 1
    
    while queue:
        u = queue.pop(0)
        for nb, idx in adj[u]:
            if idx <= K and not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    
    return count == N

def find_hamilton_cycle(K, time_limit=30.0, start_time=None):
    """
    Find a Hamilton cycle using edges with index <= K.
    Returns the cycle as a list of vertices, or None.
    """
    if start_time is None:
        start_time = time.time()
    
    # Build masks for edges <= K
    masks = [0] * N
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    
    # Check connectivity first
    if not is_connected(K):
        return None
    
    full_mask = (1 << N) - 1
    start = 0
    start_mask = 1 << start
    
    # Precompute neighbor lists filtered to edges <= K
    neighbors = [[] for _ in range(N)]
    for v in range(N):
        for nb, idx in adj[v]:
            if idx <= K:
                neighbors[v].append(nb)
            else:
                break
    
    # Use iterative DFS with backtracking
    # State: current vertex, visited mask, path
    # We fix start = 0
    
    def dfs(v, visited, path):
        if time.time() - start_time > time_limit:
            return None
        
        if len(path) == N:
            # Check if v connects back to start
            if masks[v] & (1 << start):
                return path[:]
            return None
        
        # Pruning: Check if unvisited vertices are connected to v
        unvisited_mask = full_mask ^ visited
        if unvisited_mask:
            # BFS from v to check connectivity of unvisited vertices
            reachable = 1 << v
            queue = [v]
            while queue:
                u = queue.pop(0)
                m = masks[u] & unvisited_mask
                while m:
                    lowbit = m & -m
                    nb = lowbit.bit_length() - 1
                    m ^= lowbit
                    if not (reachable & lowbit):
                        reachable |= lowbit
                        queue.append(nb)
            
            if reachable != (unvisited_mask | (1 << v)):
                return None
        
        # Try neighbors in order of edge index
        for nb in neighbors[v]:
            if visited & (1 << nb):
                continue
            
            # Additional pruning: if nb is the last unvisited vertex, 
            # it must connect back to start
            remaining_after = unvisited_mask & ~(1 << nb)
            if remaining_after == 0:
                if not (masks[nb] & (1 << start)):
                    continue
            
            result = dfs(nb, visited | (1 << nb), path + [nb])
            if result is not None:
                return result
        
        return None
    
    result = dfs(start, start_mask, [start])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s buffer for 90s limit
    
    # Binary search for min K
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    best_K = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        
        # Quick connectivity check
        if not is_connected(mid):
            lo = mid + 1
            continue
        
        # Try to find Hamilton cycle
        cycle = find_hamilton_cycle(mid, time_limit=20.0, start_time=start_time)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            # Connected but no Hamilton cycle found (or timed out)
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: Linear search upwards from 24
    for K in range(24, len(EDGE_ORDER) + 1):
        if time.time() > deadline:
            break
        if is_connected(K):
            cycle = find_hamilton_cycle(K, time_limit=10.0, start_time=start_time)
            if cycle is not None:
                return cycle
    
    # Last resort: Identity permutation
    return list(range(N))
