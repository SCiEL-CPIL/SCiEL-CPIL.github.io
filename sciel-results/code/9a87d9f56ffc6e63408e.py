"""Find a Hamilton cycle with minimized max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import sys

_N = 24

# Build the edge arrival lookup from the pinned edge order
_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# edge_arrival[u][v] = 1-based index in edge_order
_edge_arrival = [[0] * _N for _ in range(_N)]
for idx, (u, v) in enumerate(_EDGE_ORDER, 1):
    _edge_arrival[u][v] = idx
    _edge_arrival[v][u] = idx


def _cycle_score(cycle):
    """Score = max one-based arrival index of edges in the cycle (with closing edge)."""
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if _edge_arrival[u][v] > mx:
            mx = _edge_arrival[u][v]
    return mx


def _build_adj(k):
    """Build adjacency list and mask using only edges with arrival index <= k."""
    adj = [[] for _ in range(_N)]
    adj_mask = [0] * _N
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i][j] <= k:
                adj[i].append(j)
                adj[j].append(i)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj, adj_mask


def _check_prefix_feasible(k):
    """Quick feasibility check: min degree >= 2 and connected."""
    adj, _ = _build_adj(k)
    for i in range(_N):
        if len(adj[i]) < 2:
            return False
    # BFS connectivity
    visited = [False] * _N
    visited[0] = True
    stack = [0]
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    return count == _N


def _find_hamilton_cycle_exact(k, deadline):
    """
    Exact DFS with strong pruning to find a Hamilton cycle using edges <= k.
    Returns the cycle as a list of vertices, or None if not found within deadline.
    """
    adj, adj_mask = _build_adj(k)
    FULL = (1 << _N) - 1
    
    # Precompute neighbor lists sorted by degree (for Warnsdorff-like ordering)
    deg = [len(adj[i]) for i in range(_N)]
    sorted_neighbors = []
    for i in range(_N):
        sorted_neighbors.append(sorted(adj[i], key=lambda x: deg[x]))
    
    # Iterative DFS with pruning
    # State: (current, visited_mask, path)
    # We'll use a stack-based approach
    
    path = [0]
    visited_mask = 1
    current = 0
    
    # Stack of (current, visited_mask, next_index_into_sorted_neighbors)
    # Actually, let's use a recursive-like iterative approach with explicit stack
    
    # For efficiency, let's use a different approach:
    # Keep the path, and at each step try neighbors in order.
    # Use a stack of (node, visited_mask, neighbor_index)
    
    # Let's implement with explicit backtracking stack
    # Each stack entry: (node, visited_mask, path_so_far, neighbor_idx)
    
    stack = [(0, 1, [0], 0)]
    
    while stack:
        if time.time() > deadline:
            return None
        
        node, v_mask, p, nidx = stack[-1]
        
        if len(p) == _N:
            # Check if we can close the cycle
            if _edge_arrival[node][0] <= k:
                return p
            stack.pop()
            continue
        
        # Try neighbors starting from nidx
        found = False
        while nidx < len(sorted_neighbors[node]):
            nidx += 1
            v = sorted_neighbors[node][nidx - 1]
            
            if v_mask & (1 << v):
                continue
            
            # If this is the last vertex to add, it must connect to 0
            if len(p) == _N - 1:
                if _edge_arrival[v][0] > k:
                    continue
            
            new_mask = v_mask | (1 << v)
            new_remaining = FULL & ~new_mask
            
            # Pruning: check if any unvisited vertex becomes isolated
            # or has insufficient connections
            valid = True
            r = new_remaining
            while r:
                bit = r & -r
                u = bit.bit_length() - 1
                r -= bit
                # u is unvisited
                u_unvisited = adj_mask[u] & new_remaining
                if u_unvisited == 0:
                    # u must have >= 2 neighbors in visited set
                    u_visited = adj_mask[u] & new_mask
                    if bin(u_visited).count('1') < 2:
                        valid = False
                        break
                elif bin(u_unvisited).count('1') == 1:
                    # u has exactly one unvisited neighbor
                    # It must also have at least one visited neighbor (to form cycle)
                    u_visited = adj_mask[u] & new_mask
                    if bin(u_visited).count('1') < 1:
                        valid = False
                        break
            if not valid:
                continue
            
            # Commit
            p.append(v)
            stack[-1] = (node, v_mask, p, nidx)
            stack.append((v, new_mask, p, 0))
            found = True
            break
        
        if not found:
            # Backtrack
            stack.pop()
            if stack:
                # Remove v from path
                p.pop()
    
    return None


def _find_hamilton_cycle_iterative(k, deadline):
    """
    Alternative: use a more efficient iterative approach with bitmasks.
    """
    adj, adj_mask = _build_adj(k)
    FULL = (1 << _N) - 1
    
    # Precompute for each vertex, the list of neighbors
    # and use Warnsdorff's rule with backtracking
    
    # We'll use a stack-based DFS
    # State: (current, visited_mask, path)
    
    # Start from vertex 0
    path = [0]
    visited_mask = 1
    current = 0
    
    # Stack for backtracking: each entry is (node, visited_mask, path_copy, next_neighbor_idx)
    # To save memory, we'll track the path separately
    
    # Let's use a simpler approach: recursive with sys.setrecursionlimit
    # But for N=24, we need iterative
    
    # Iterative DFS with explicit stack
    # Stack entry: (node, visited_mask, path_length, next_neighbor_idx)
    
    stack = [(0, 1, 1, 0)]  # (node, visited_mask, path_len, next_idx)
    path = [0]  # path[0] = 0, path[path_len-1] = current
    
    while stack:
        if time.time() > deadline:
            return None
        
        node, v_mask, plen, nidx = stack[-1]
        
        if plen == _N:
            if _edge_arrival[node][0] <= k:
                return path[:]
            stack.pop()
            continue
        
        # Find next neighbor to try
        found = False
        neighbors = adj[node]
        while nidx < len(neighbors):
            nidx += 1
            v = neighbors[nidx - 1]
            
            if v_mask & (1 << v):
                continue
            
            # Last vertex must connect to 0
            if plen == _N - 1:
                if _edge_arrival[v][0] > k:
                    continue
            
            new_mask = v_mask | (1 << v)
            new_remaining = FULL & ~new_mask
            
            # Pruning
            valid = True
            r = new_remaining
            while r:
                bit = r & -r
                u = bit.bit_length() - 1
                r -= bit
                u_unvisited = adj_mask[u] & new_remaining
                if u_unvisited == 0:
                    u_visited = adj_mask[u] & new_mask
                    if bin(u_visited).count('1') < 2:
                        valid = False
                        break
            if not valid:
                continue
            
            # Commit
            path.append(v)
            stack[-1] = (node, v_mask, plen, nidx)
            stack.append((v, new_mask, plen + 1, 0))
            found = True
            break
        
        if not found:
            stack.pop()
            if plen > 1:
                path.pop()
    
    return None


def _local_search_2opt(cycle, deadline):
    """Improve a Hamilton cycle using 2-opt moves to minimize max edge arrival."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    # Baseline
    best_cycle = list(range(_N))
    best_score = _cycle_score(best_cycle)
    
    # Binary search for minimum k
    lo, hi = 2, 276
    found_k = None
    search_deadline = start_time + 70.0
    
    while lo <= hi and time.time() < search_deadline:
        mid = (lo + hi) // 2
        
        if not _check_prefix_feasible(mid):
            lo = mid + 1
            continue
        
        # Try exact DFS with time limit
        time_left = search_deadline - time.time()
        if time_left < 1.0:
            break
        
        cycle = _find_hamilton_cycle_exact(mid, start_time + time_left * 0.8)
        
        if cycle is not None:
            found_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If found_k, try to get the cycle with more time
    if found_k is not None:
        cycle_deadline = min(deadline - 5.0, start_time + 80.0)
        cycle = _find_hamilton_cycle_exact(found_k, cycle_deadline)
        if cycle is not None:
            score = _cycle_score(cycle)
            if score < best_score:
                best_cycle = cycle
                best_score = score
    
    # Local search to polish
    if time.time() < deadline - 2.0:
        polished, polished_score = _local_search_2opt(best_cycle, deadline - 1.0)
        if polished_score < best_score:
            best_cycle = polished
            best_score = polished_score
    
    return best_cycle

# EVOLVE-BLOCK-END
