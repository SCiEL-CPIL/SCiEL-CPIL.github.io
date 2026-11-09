import time
import sys

# EVOLVE-BLOCK-START
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

TOTAL_EDGES = N * (N - 1) // 2

# Precompute edge position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = (u, v) if u < v else (v, u)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = (u, v) if u < v else (v, u)
    return edge_pos[(a, b)]

def build_adj_bitsets(threshold):
    """Build adjacency as list of bitmasks for edges with index <= threshold."""
    adj = [0] * N
    for i in range(threshold):
        u, v = EDGE_ORDER[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def popcount(x):
    return bin(x).count('1')

def is_connected(adj, n):
    """Check if graph is connected using BFS with bitmasks."""
    visited = 1  # start from vertex 0
    stack = [0]
    while stack:
        u = stack.pop()
        # Find unvisited neighbors
        neighbors = adj[u] & ~visited
        while neighbors:
            lsb = neighbors & -neighbors
            v = lsb.bit_length() - 1
            visited |= lsb
            stack.append(v)
            neighbors ^= lsb
    return visited == (1 << n) - 1

def has_ham_cycle_fast(adj, n, start=0, deadline=None):
    """
    Fast Hamilton cycle check using DFS with bitmasks and aggressive pruning.
    Returns path list or None.
    """
    full_mask = (1 << n) - 1
    path = [start]
    visited = 1 << start
    max_depth = n - 1
    
    # Precompute degrees for ordering
    def dfs(current, visited, depth):
        if deadline and time.time() > deadline:
            return None
        
        if depth == max_depth:
            # All vertices visited, check if current connects to start
            if adj[current] & (1 << start):
                return path[:]
            return None
        
        # Get unvisited neighbors
        neighbors_mask = adj[current] & ~visited
        
        if neighbors_mask == 0:
            return None
        
        # Collect and sort neighbors by their degree (fewest unvisited first - "fail fast")
        neighbors = []
        temp = neighbors_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Count how many unvisited neighbors v has (excluding current which is visited)
            v_unvisited = adj[v] & ~visited
            deg = popcount(v_unvisited)
            neighbors.append((deg, v))
            temp ^= lsb
        
        neighbors.sort()
        
        for deg, v in neighbors:
            new_visited = visited | (1 << v)
            
            # Pruning 1: if v is not the last vertex, v must have at least one 
            # unvisited neighbor remaining (otherwise we get stuck)
            if depth + 1 < max_depth:
                if (adj[v] & ~new_visited) == 0:
                    continue
            
            # Pruning 2: check that start still has at least 2 neighbors that are 
            # either visited (in path) or will be (unvisited) - actually, start needs
            # at least one unvisited neighbor OR the last vertex must connect to start
            # This is complex; skip for now
            
            path.append(v)
            result = dfs(v, new_visited, depth + 1)
            if result is not None:
                return result
            path.pop()
            
        return None
    
    return dfs(start, visited, 0)

def has_ham_cycle_multi(adj, n, deadline=None):
    """Try multiple starting vertices and return best path found."""
    best = None
    for start in range(n):
        if deadline and time.time() > deadline:
            break
        result = has_ham_cycle_fast(adj, n, start=start, deadline=deadline)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_path = None
    best_T = TOTAL_EDGES + 1
    
    # Binary search for the minimum threshold T where a Hamilton cycle exists
    lo = N  # Minimum edges in a cycle
    hi = TOTAL_EDGES
    
    # First, verify we can find a cycle at hi
    adj_hi = build_adj_bitsets(hi)
    path_hi = has_ham_cycle_multi(adj_hi, N, deadline=deadline)
    if path_hi is not None:
        best_path = path_hi
        best_T = hi
    
    while lo <= hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        if mid >= best_T:
            break
        
        adj_mid = build_adj_bitsets(mid)
        
        # Quick checks before expensive DFS
        # 1. All vertices must have degree >= 2
        valid = True
        for v in range(N):
            if popcount(adj_mid[v]) < 2:
                valid = False
                break
        if not valid:
            lo = mid + 1
            continue
        
        # 2. Graph must be connected
        if not is_connected(adj_mid, N):
            lo = mid + 1
            continue
        
        # 3. Try to find Hamilton cycle
        path = has_ham_cycle_multi(adj_mid, N, deadline=deadline)
        
        if path is not None:
            best_path = path
            best_T = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a cycle, try to improve it by local search
    if best_path is not None and time.time() < deadline - 10:
        # Try to find a cycle with lower max edge index by trying nearby thresholds
        for T in range(best_T - 1, max(N - 1, best_T - 50), -1):
            if time.time() > deadline:
                break
            if T >= best_T:
                continue
            
            adj_T = build_adj_bitsets(T)
            
            # Quick checks
            valid = True
            for v in range(N):
                if popcount(adj_T[v]) < 2:
                    valid = False
                    break
            if not valid:
                continue
            if not is_connected(adj_T, N):
                continue
            
            path = has_ham_cycle_multi(adj_T, N, deadline=deadline)
            if path is not None:
                best_path = path
                best_T = T
                break  # Found lower, stop
    
    if best_path is None:
        # Fallback: should not happen, but return something valid
        return list(range(N))
    
    return best_path
# EVOLVE-BLOCK-END
