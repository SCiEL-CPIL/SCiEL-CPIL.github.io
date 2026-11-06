import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

# Build position map: edge -> arrival index (1-based)
_edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = _edge_key(u, v)
    _edge_pos[key] = i + 1

# Precompute weights for Held-Karp (0-based index for convenience, or 1-based doesn't matter for comparison)
# Weight is the arrival index. We want to minimize the sum of weights, which correlates with minimizing max weight.
weights = [[0] * N for _ in range(N)]
for u in range(N):
    for v in range(N):
        if u == v:
            continue
        weights[u][v] = _edge_pos.get(_edge_key(u, v), float('inf'))

def _compute_lower_bound():
    deg_counts = [0] * N
    last_edge_idx = [0] * N
    for i, (u, v) in enumerate(EDGE_ORDER):
        idx = i + 1
        if deg_counts[u] == 1:
            last_edge_idx[u] = idx
        deg_counts[u] += 1
        
        if deg_counts[v] == 1:
            last_edge_idx[v] = idx
        deg_counts[v] += 1
        
    max_idx = 0
    for v in range(N):
        if last_edge_idx[v] > max_idx:
            max_idx = last_edge_idx[v]
    return max_idx

MIN_K_LOWER_BOUND = _compute_lower_bound()

def _cycle_score(cycle):
    """Compute the score (max arrival index) for a Hamilton cycle."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        key = _edge_key(u, v)
        idx = _edge_pos.get(key, float('inf'))
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _two_opt_improve(cycle, edge_pos, max_iter=10000):
    """Try to improve a cycle's score using 2-opt and Or-opt moves."""
    n = len(cycle)
    best_score = _cycle_score(cycle)
    
    # Helper to get score quickly
    def get_score(cyc):
        m = 0
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            idx = edge_pos.get(_edge_key(u, v), float('inf'))
            if idx > m: m = idx
        return m

    for _ in range(max_iter):
        improved = False
        
        # 2-Opt
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                score = get_score(new_cycle)
                if score < best_score:
                    cycle = new_cycle
                    best_score = score
                    improved = True
                    
        # Or-Opt (move single vertex)
        for i in range(n):
            if i == 0: continue 
            v = cycle[i]
            new_cycle = cycle[:i] + cycle[i+1:]
            for j in range(len(new_cycle)):
                candidate = new_cycle[:j] + [v] + new_cycle[j:]
                score = get_score(candidate)
                if score < best_score:
                    cycle = candidate
                    best_score = score
                    improved = True
                    break # Restart loop if improved
            
        if not improved:
            break
    return cycle, best_score

def _held_karp():
    """
    Solve Minimum Weight Hamiltonian Cycle using Held-Karp DP.
    Returns the optimal cycle (list of vertices) and its max edge weight.
    """
    # dp[mask][last] = minimum weight of a path starting at 0, visiting set 'mask', ending at 'last'
    # mask includes 0 and last.
    # We fix start = 0.
    INF = float('inf')
    size = 1 << N
    dp = [[INF] * N for _ in range(size)]
    parent = [[-1] * N for _ in range(size)]
    
    dp[1][0] = 0
    
    for mask in range(size):
        if not (mask & 1): # Must include start node 0
            continue
        for last in range(N):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue
            
            # Try to extend path to next
            # next must not be in mask
            # Optimization: iterate over bits not in mask
            remaining = ((1 << N) - 1) ^ mask
            while remaining:
                lowbit = remaining & -remaining
                nxt = lowbit.bit_length() - 1
                remaining ^= lowbit
                
                new_mask = mask | (1 << nxt)
                new_cost = dp[mask][last] + weights[last][nxt]
                
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    full_mask = (1 << N) - 1
    best_last = -1
    best_cost = INF
    
    for last in range(1, N):
        # Cost to close the cycle
        total_cost = dp[full_mask][last] + weights[last][0]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last = last
            
    if best_last == -1:
        return None, INF

    # Reconstruct path
    path = []
    mask = full_mask
    cur = best_last
    while cur != -1:
        path.append(cur)
        prev = parent[mask][cur]
        if prev == -1:
            break
        mask ^= (1 << cur)
        cur = prev
        
    path.reverse()
    # path should be [0, ..., best_last]
    # Check if it's valid
    if len(path) != N or path[0] != 0:
        # Fallback if reconstruction failed (shouldn't happen)
        return list(range(N)), _cycle_score(list(range(N)))
        
    return path, best_cost

def _warnsdorff_dfs(adj, start, max_nodes=5000000):
    """DFS with Warnsdorff's heuristic."""
    n = N
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    def dfs(path):
        if len(path) == n:
            if path[-1] in adj[path[0]]:
                return True
            return False
        
        current = path[-1]
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return False
        
        neighbors.sort(key=lambda v: len([w for w in adj[v] if not visited[w]]))
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            if dfs(path):
                return True
            path.pop()
            visited[v] = False
        return False
    
    if dfs(path):
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s margin
    
    best_cycle = None
    best_score = float('inf')
    
    # 1. Try Held-Karp (Exact Min-Sum)
    # This is O(N^2 * 2^N) which is ~ 24^2 * 16M = 900M ops. In Python this is too slow.
    # 16M states * 24 neighbors = 400M transitions. Each transition is a dict lookup or list access.
    # It will likely take > 60s in pure Python.
    # Let's check if we can optimize or skip.
    # Actually, N=24 is the limit for Held-Karp in C++. In Python, it's usually N<=20.
    # So we should probably NOT run full Held-Karp in Python within 90s.
    # We will use a heuristic based on the "Min-Sum" idea but faster, or just stick to the current good heuristic.
    # However, the directive says "Formulate the problem as a minimum-weight Hamiltonian cycle... and solve it exactly using Held-Karp".
    # If I can't solve it exactly in time, I should use the best approximation or a smaller exact solve?
    # No, the directive is a strategy to TRY. If it's too slow, I should adapt.
    # Let's try a limited Held-Karp or a very optimized version?
    # Or maybe use the heuristic from the current program which already scored 5.93.
    # The current program uses binary search on K + Warnsdorff + 2-opt.
    # The "Min-Sum" correlation suggests that finding a cycle with low sum of indices is good.
    # We can use a TSP heuristic (Nearest Neighbor, 2-Opt) on the weight matrix to find a low-sum cycle,
    # then check its max-index score.
    
    # Let's implement a fast TSP heuristic:
    # 1. Greedy construction (Nearest Neighbor or similar)
    # 2. 2-Opt improvement on the Sum-Weight metric.
    # 3. Evaluate the Max-Index score.
    
    # This is structurally different from "Binary Search on K + Hamiltonian Cycle Search".
    # It directly optimizes the proxy objective.
    
    # Step 1: Greedy Construction
    # Start at 0, go to neighbor with smallest weight, etc.
    # Weights are arrival indices.
    
    def greedy_tsp():
        visited = [False] * N
        visited[0] = True
        path = [0]
        for _ in range(N - 1):
            last = path[-1]
            best_next = -1
            best_w = INF
            for v in range(N):
                if not visited[v]:
                    w = weights[last][v]
                    if w < best_w:
                        best_w = w
                        best_next = v
            if best_next == -1:
                break
            visited[best_next] = True
            path.append(best_next)
        return path

    def two_opt_tsp(cycle):
        """Improve cycle to minimize SUM of weights."""
        n = len(cycle)
        if n < 3: return cycle
        
        def get_sum(cyc):
            s = 0
            for i in range(n):
                s += weights[cyc[i]][cyc[(i+1)%n]]
            return s
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if j == i + 1 or (i == 0 and j == n - 1):
                        continue
                    # Reverse segment i+1 to j
                    # Old edges: (i, i+1) and (j, j+1)
                    # New edges: (i, j) and (i+1, j+1)
                    # Note: indices are modulo n
                    j_next = (j + 1) % n
                    i_next = (i + 1) % n
                    
                    old_w = weights[cycle[i]][cycle[i_next]] + weights[cycle[j]][cycle[j_next]]
                    new_w = weights[cycle[i]][cycle[j]] + weights[cycle[i_next]][cycle[j_next]]
                    
                    if new_w < old_w:
                        # Perform reversal
                        # cycle[i+1 ... j] reversed
                        cycle[i+1:j+1] = cycle[i+1:j+1][::-1]
                        improved = True
        return cycle

    # Run Greedy TSP
    greedy_cycle = greedy_tsp()
    # Check if it's a valid cycle (it is, by construction, but check closing edge weight is finite)
    if weights[greedy_cycle[-1]][greedy_cycle[0]] < INF:
        # Improve with 2-Opt on Sum
        improved_cycle = two_opt_tsp(greedy_cycle)
        score = _cycle_score(improved_cycle)
        if score < best_score:
            best_score = score
            best_cycle = improved_cycle
            
    # Also try the original heuristic (Binary Search) if time permits?
    # The original heuristic was quite good (5.93).
    # Let's see if we have time.
    
    # Step 2: Try the original Binary Search approach if time allows
    # This acts as a backup or a second diversification.
    if time.time() - start_time < deadline - 10:
        # Re-use the logic from the current program
        # Build adjacency for a given K
        def _build_adjacency(K):
            adj = [set() for _ in range(N)]
            for i, (u, v) in enumerate(EDGE_ORDER):
                if i + 1 > K:
                    break
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def _find_hamilton_cycle(K, time_limit=2.0):
            adj = _build_adjacency(K)
            for v in range(N):
                if len(adj[v]) < 2:
                    return None
            
            start_time_inner = time.time()
            # Try Warnsdorff from each starting vertex
            for s in range(N):
                if time.time() - start_time_inner > time_limit:
                    break
                # Simple Warnsdorff
                path = [s]
                visited = [False] * N
                visited[s] = True
                stack = [(s, path, visited)]
                # Actually, let's just use a simple DFS with Warnsdorff ordering
                # To avoid recursion depth issues and overhead, let's do iterative or simple recursive with limit
                
                def dfs_warnsdorff(cur_path, cur_visited):
                    if len(cur_path) == N:
                        if cur_path[-1] in adj[cur_path[0]]:
                            return cur_path
                        return None
                    
                    current = cur_path[-1]
                    neighbors = [v for v in adj[current] if not cur_visited[v]]
                    if not neighbors:
                        return None
                    
                    neighbors.sort(key=lambda v: len([w for w in adj[v] if not cur_visited[w]]))
                    
                    for v in neighbors:
                        new_visited = cur_visited.copy()
                        new_visited[v] = True
                        res = dfs_warnsdorff(cur_path + [v], new_visited)
                        if res:
                            return res
                    return None
                
                cycle = dfs_warnsdorff([s], [False]*N)
                if cycle is not None:
                    return cycle
            return None

        lo = MIN_K_LOWER_BOUND
        hi = 276
        best_cycle_bs = None
        best_K = hi
        
        # Binary search
        while lo < hi and (time.time() - start_time) < deadline - 15:
            mid = (lo + hi) // 2
            cycle = _find_hamilton_cycle(mid, time_limit=1.0)
            if cycle is not None:
                best_cycle_bs = cycle
                best_K = mid
                hi = mid - 1
            else:
                lo = mid + 1
                
        if best_cycle_bs is not None:
            # Refine with 2-opt on Max-Index
            refined, refined_score = _two_opt_improve(best_cycle_bs, _edge_pos, max_iter=5000)
            if refined_score < best_score:
                best_score = refined_score
                best_cycle = refined

    if best_cycle is None:
        return list(range(N))
        
    return best_cycle

# EVOLVE-BLOCK-END
