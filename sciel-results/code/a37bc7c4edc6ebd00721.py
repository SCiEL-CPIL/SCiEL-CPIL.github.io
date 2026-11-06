"""Find Hamilton cycle with minimal max edge arrival index."""
import time
import sys

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position lookup
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[frozenset((u, v))] = i

def build_adj(K):
    """Build adjacency list using only edges with position <= K, sorted by position."""
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        adj[u].append((v, i))
        adj[v].append((u, i))
    for v in range(N):
        adj[v].sort(key=lambda x: x[1])
    return adj

def is_connected(adj):
    """Check if the graph is connected and all vertices have degree >= 2."""
    # Check degree constraint first
    for v in range(N):
        if len(adj[v]) < 2:
            return False
    # Check connectivity
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        v = stack.pop()
        count += 1
        for nb, _ in adj[v]:
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)
    return count == N

def find_hamilton_cycle(K, time_limit=None, deadline=None):
    """Find a Hamilton cycle using only edges with position <= K.
    Returns the cycle as a list of vertices, or None."""
    adj = build_adj(K)
    
    # Quick feasibility check
    if not is_connected(adj):
        return None
    
    sys.setrecursionlimit(10000)
    full_mask = (1 << N) - 1
    
    # Precompute neighbor lists for faster access
    adj_list = [[] for _ in range(N)]
    for v in range(N):
        adj_list[v] = [nb for nb, _ in adj[v]]
    
    # Try multiple starting vertices, prioritized by degree
    starts = sorted(range(N), key=lambda v: len(adj[v]))
    
    for start in starts:
        if deadline and time.time() > deadline:
            return None
        if time_limit and time.time() > time_limit:
            return None
        start_bit = 1 << start
        
        # Check if start can connect back to itself (needs degree >= 2)
        if len(adj[start]) < 2:
            continue
        
        def dfs(v, visited_mask, path):
            if time_limit and time.time() > time_limit:
                return None
            if deadline and time.time() > deadline:
                return None
            if len(path) == N:
                # Check if v connects back to start
                for nb, _ in adj[v]:
                    if nb == start:
                        return path[:]
                return None
            
            # Warnsdorff: try neighbors with fewest unvisited options first
            candidates = []
            for nb in adj_list[v]:
                if not (visited_mask & (1 << nb)):
                    # Count unvisited neighbors of nb
                    cnt = 0
                    for nb2 in adj_list[nb]:
                        if not (visited_mask & (1 << nb2)):
                            cnt += 1
                    candidates.append((cnt, nb))
            candidates.sort()
            for cnt, nb in candidates:
                result = dfs(nb, visited_mask | (1 << nb), path + [nb])
                if result is not None:
                    return result
            return None
        
        result = dfs(start, start_bit, [start])
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (60s kill)
    
    # Binary search for minimum K where Hamilton cycle exists
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    
    while lo <= hi:
        if time.time() > deadline - 3:
            break
        mid = (lo + hi) // 2
        # Allocate more time for small K (harder to find), less for large K (easier)
        if mid < 100:
            time_budget = min(3.0, max(1.0, deadline - 3 - time.time()))
        elif mid < 200:
            time_budget = min(2.0, max(0.5, deadline - 3 - time.time()))
        else:
            time_budget = min(1.0, max(0.3, deadline - 3 - time.time()))
        
        cycle = find_hamilton_cycle(mid, time_limit=start_time + time_budget, deadline=deadline)
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_cycle is not None:
        # Try to improve: search slightly below the found K with more time
        if hi >= 24 and time.time() < deadline - 2:
            for K in range(hi, max(23, hi - 20), -1):
                if time.time() > deadline - 1:
                    break
                time_budget = min(2.0, max(0.3, deadline - 1 - time.time()))
                cycle = find_hamilton_cycle(K, time_limit=start_time + time_budget, deadline=deadline)
                if cycle is not None:
                    best_cycle = cycle
                    break
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback: try larger K values directly
    for K in range(100, len(EDGE_ORDER) + 1, 5):
        if time.time() > deadline - 1:
            break
        cycle = find_hamilton_cycle(K, time_limit=start_time + 1.0, deadline=deadline)
        if cycle is not None:
            best_cycle = cycle
            break
    
    if best_cycle is not None:
        return best_cycle
    
    # Last resort: identity permutation
    return list(range(N))
