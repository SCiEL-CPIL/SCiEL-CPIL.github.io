"""Find a Hamilton cycle with minimized max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import random
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
    """Build adjacency list using only edges with arrival index <= k."""
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i][j] <= k:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _has_hamilton_cycle_bfs(k, timeout=2.0):
    """
    Check if a Hamilton cycle exists using only edges with arrival index <= k.
    Uses iterative DFS with strong pruning:
    - Degree check: every vertex must have degree >= 2 in the prefix graph
    - Connectivity check: the prefix graph must be connected
    - During search: forced moves (degree-2 vertices), articulation point pruning
    """
    adj = _build_adj(k)
    
    # Check minimum degree
    for i in range(_N):
        if len(adj[i]) < 2:
            return False
    
    # Check connectivity via BFS from vertex 0
    visited = [False] * _N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count < _N:
        return False
    
    # Iterative DFS with backtracking
    # State: (current, visited_mask, path)
    # We fix vertex 0 as the start.
    FULL = (1 << _N) - 1
    
    # Precompute neighbor masks for fast intersection
    adj_mask = [0] * _N
    for i in range(_N):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # Use a stack-based DFS
    # Each stack entry: (current, visited_mask, path_tuple, next_neighbor_idx)
    # To save memory, we track the path implicitly via the stack
    
    sys.setrecursionlimit(10000)
    
    def dfs_iter():
        # Stack of (current, visited, path_list_copy, neighbor_index)
        # We'll use a simpler recursive approach with pruning but with a time limit
        pass
    
    # Let's use a recursive DFS with careful pruning
    # But with a time limit to avoid hanging
    
    start_time = time.time()
    
    # Order neighbors by degree (ascending) for better pruning
    adj_sorted = []
    for i in range(_N):
        adj_sorted.append(sorted(adj[i], key=lambda x: len(adj[x])))
    
    def dfs(current, visited, path_len, path):
        if time.time() - start_time > timeout:
            return False
        
        if path_len == _N:
            # Check if we can close the cycle: edge from current to 0
            if _edge_arrival[current][0] <= k:
                return True
            return False
        
        # Determine which neighbors we can visit
        # Neighbors must be unvisited, and if path_len == N-1, must connect back to 0
        candidates = []
        for nxt in adj_sorted[current]:
            if visited & (1 << nxt):
                continue
            if path_len == _N - 1:
                # Last vertex: must connect back to 0
                if _edge_arrival[nxt][0] <= k:
                    candidates.append(nxt)
            else:
                candidates.append(nxt)
        
        # Pruning: check if any unvisited vertex (except 0 which is already visited)
        # would become isolated if we don't visit it soon
        remaining = FULL & ~visited
        rem_count = bin(remaining).count('1')
        
        for nxt in candidates:
            new_visited = visited | (1 << nxt)
            
            # Pruning: after visiting nxt, check that all unvisited vertices 
            # (excluding 0) still have at least one neighbor that is either
            # unvisited or is the current endpoint (for eventual connection)
            if rem_count > 1:
                # Check unvisited vertices (excluding 0)
                valid = True
                r = remaining & ~(1 << nxt)
                while r:
                    v_bit = r & -r
                    v = v_bit.bit_length() - 1
                    r -= v_bit
                    if v == 0:
                        continue
                    # v must have at least one neighbor in (unvisited after nxt) or in path
                    # Simple check: v must have at least one neighbor that is not yet visited
                    # (excluding itself)
                    v_neighbors_unvisited = adj_mask[v] & (FULL & ~new_visited)
                    if v_neighbors_unvisited == 0:
                        # v has no unvisited neighbors; it must connect to the path
                        # For a Hamilton cycle, v needs degree >= 2 in the final cycle
                        # If all its neighbors are already in the path, it can only connect
                        # to at most 2 of them (its two cycle neighbors)
                        # Check if v has at least 2 neighbors in the path
                        v_neighbors_in_path = adj_mask[v] & new_visited
                        if bin(v_neighbors_in_path).count('1') < 2:
                            valid = False
                            break
                if not valid:
                    continue
            
            path.append(nxt)
            if dfs(nxt, new_visited, path_len + 1, path):
                return True
            path.pop()
        
        return False
    
    path = [0]
    return dfs(0, 1, 1, path)


def _find_cycle_in_prefix(k, timeout=5.0):
    """Find a Hamilton cycle using only edges with arrival index <= k. Returns cycle or None."""
    adj = _build_adj(k)
    
    # Check minimum degree
    for i in range(_N):
        if len(adj[i]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * _N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count < _N:
        return None
    
    FULL = (1 << _N) - 1
    adj_mask = [0] * _N
    for i in range(_N):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # Sort neighbors by degree for better pruning
    adj_sorted = []
    for i in range(_N):
        adj_sorted.append(sorted(adj[i], key=lambda x: len(adj[x])))
    
    start_time = time.time()
    path = [0]
    
    def dfs(current, visited, path_len):
        if time.time() - start_time > timeout:
            return False
        
        if path_len == _N:
            if _edge_arrival[current][0] <= k:
                return True
            return False
        
        candidates = []
        for nxt in adj_sorted[current]:
            if visited & (1 << nxt):
                continue
            if path_len == _N - 1:
                if _edge_arrival[nxt][0] <= k:
                    candidates.append(nxt)
            else:
                candidates.append(nxt)
        
        remaining = FULL & ~visited
        rem_count = bin(remaining).count('1')
        
        for nxt in candidates:
            new_visited = visited | (1 << nxt)
            
            if rem_count > 1:
                valid = True
                r = remaining & ~(1 << nxt)
                while r:
                    v_bit = r & -r
                    v = v_bit.bit_length() - 1
                    r -= v_bit
                    if v == 0:
                        continue
                    v_neighbors_unvisited = adj_mask[v] & (FULL & ~new_visited)
                    if v_neighbors_unvisited == 0:
                        v_neighbors_in_path = adj_mask[v] & new_visited
                        if bin(v_neighbors_in_path).count('1') < 2:
                            valid = False
                            break
                if not valid:
                    continue
            
            path.append(nxt)
            if dfs(nxt, new_visited, path_len + 1):
                return True
            path.pop()
        
        return False
    
    if dfs(0, 1, 1):
        return path[:]
    return None


def _local_search_2opt(cycle, max_iter=2000, deadline=None):
    """Improve a Hamilton cycle using 2-opt moves to minimize max edge arrival."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment i+1..j
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


def _greedy_construct(k):
    """Try to greedily construct a Hamilton cycle using edges <= k."""
    adj = _build_adj(k)
    
    # Check minimum degree
    for i in range(_N):
        if len(adj[i]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * _N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count < _N:
        return None
    
    # Greedy: start at 0, always go to the neighbor with fewest options
    # that hasn't been visited yet
    path = [0]
    visited_set = {0}
    
    for _ in range(_N - 1):
        current = path[-1]
        # Get unvisited neighbors
        unvisited_neighbors = [v for v in adj[current] if v not in visited_set]
        
        if not unvisited_neighbors:
            return None
        
        # If this is the last step, must connect back to 0
        if len(path) == _N - 1:
            valid = [v for v in unvisited_neighbors if _edge_arrival[v][0] <= k]
            if not valid:
                return None
            next_v = valid[0]
        else:
            # Choose the neighbor with fewest unvisited neighbors (most constrained)
            def constraint(v):
                return len([u for u in adj[v] if u not in visited_set and u != current])
            next_v = min(unvisited_neighbors, key=constraint)
        
        path.append(next_v)
        visited_set.add(next_v)
    
    # Check if cycle closes
    if _edge_arrival[path[-1]][0] <= k:
        return path
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s limit
    
    # Get baseline
    baseline = list(range(_N))
    best_cycle = baseline
    best_score = _cycle_score(baseline)
    
    # Binary search for the minimum k where a Hamilton cycle exists
    # Edge indices range from 1 to C(24,2) = 276
    lo, hi = 1, 276
    found_k = None
    
    # First, check if the full graph has a Hamilton cycle (it should, it's K24)
    # Binary search with time budget
    search_budget = 40.0  # seconds for binary search
    search_deadline = start_time + search_budget
    
    while lo <= hi and time.time() < search_deadline:
        mid = (lo + hi) // 2
        
        # Quick check: minimum degree in prefix
        min_deg = _N
        for i in range(_N):
            deg = 0
            for j in range(_N):
                if i != j and _edge_arrival[i][j] <= mid:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
        if min_deg < 2:
            lo = mid + 1
            continue
        
        # Try to find a cycle
        result = _has_hamilton_cycle_bfs(mid, timeout=min(3.0, max(0.5, search_deadline - time.time())))
        
        if result:
            found_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search found a k, try to find an actual cycle
    if found_k is not None:
        # Try to find a cycle at found_k
        cycle_deadline = min(deadline - 10.0, start_time + 60.0)
        
        for attempt in range(10):
            if time.time() > cycle_deadline:
                break
            timeout = min(5.0, max(0.5, cycle_deadline - time.time()))
            cycle = _find_cycle_in_prefix(found_k, timeout=timeout)
            if cycle is not None:
                score = _cycle_score(cycle)
                if score < best_score:
                    best_cycle = cycle
                    best_score = score
                    break
        
        # Also try greedy at found_k
        if time.time() < cycle_deadline:
            cycle = _greedy_construct(found_k)
            if cycle is not None:
                score = _cycle_score(cycle)
                if score < best_score:
                    best_cycle = cycle
                    best_score = score
    
    # If we haven't made good progress, try local search from random starts
    if best_score > 100 and time.time() < deadline - 5.0:
        rng = random.Random(42)
        ls_deadline = deadline - 2.0
        
        for restart in range(20):
            if time.time() > ls_deadline:
                break
            
            # Random permutation
            perm = list(range(_N))
            rng.shuffle(perm)
            
            # Local search
            improved_cycle, improved_score = _local_search_2opt(
                perm, max_iter=500, deadline=min(ls_deadline, time.time() + 3.0)
            )
            
            if improved_score < best_score:
                best_cycle = improved_cycle
                best_score = improved_score
    
    # Final local search on the best cycle found
    if time.time() < deadline - 1.0:
        final_cycle, final_score = _local_search_2opt(
            best_cycle, max_iter=1000, deadline=deadline - 0.5
        )
        if final_score < best_score:
            best_cycle = final_cycle
            best_score = final_score
    
    return best_cycle

# EVOLVE-BLOCK-END
