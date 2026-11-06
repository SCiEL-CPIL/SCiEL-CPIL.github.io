import time
import random

# EVOLVE-BLOCK-START
# Precompute edge arrival index for O(1) lookup
_EDGE_INDEX = {}
for _i, (_u, _v) in enumerate([
    [3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23],
    [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19],
    [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19],
    [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8],
    [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17],
    [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23],
    [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4],
    [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12],
    [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20],
    [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16],
    [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18],
    [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21],
    [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14],
    [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6],
    [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8],
    [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18],
    [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12],
    [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23],
    [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20],
    [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5],
    [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11],
    [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7],
    [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9],
    [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4],
    [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14],
    [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23],
    [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16],
    [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]
]):
    _EDGE_INDEX[frozenset((_u, _v))] = _i + 1  # one-based

N = 24
TIME_LIMIT = 50.0  # Safety margin before 60s limit

def _edge_idx(u, v):
    """Get one-based arrival index of edge (u,v)"""
    return _EDGE_INDEX[frozenset((u, v))]

def _cycle_score(cycle):
    """Compute max edge arrival index for a Hamilton cycle"""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = _edge_idx(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _two_opt_improve(cycle, time_budget):
    """
    Fast 2-opt: only accept moves that strictly reduce the max edge index.
    Uses incremental tracking to avoid full recomputation.
    """
    n = len(cycle)
    best_score = _cycle_score(cycle)
    
    # Precompute all edge indices
    edge_indices = []
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edge_indices.append(_edge_idx(u, v))
    
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        for i in range(n - 1):
            if time.time() >= time_budget:
                return cycle, best_score
            for j in range(i + 1, n - 1):
                # 2-opt: reverse segment from i+1 to j
                u1 = cycle[i]
                v1 = cycle[i + 1]
                u2 = cycle[j]
                v2 = cycle[(j + 1) % n]
                
                new_e1 = _edge_idx(u1, u2)
                new_e2 = _edge_idx(v1, v2)
                
                # Quick reject: if either new edge exceeds current best, skip
                if new_e1 > best_score or new_e2 > best_score:
                    continue
                
                # Both new edges <= best_score. Check if removing old edges reduces max.
                old_e1 = edge_indices[i]
                old_e2 = edge_indices[j]
                
                # If neither old edge was the unique max, removing them doesn't help
                # unless one of them equals best_score and it's the only occurrence
                if old_e1 < best_score and old_e2 < best_score:
                    continue
                
                # Compute new max: max of all edges except positions i and j, plus new_e1, new_e2
                new_max = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if edge_indices[k] > new_max:
                        new_max = edge_indices[k]
                
                if new_e1 > new_max:
                    new_max = new_e1
                if new_e2 > new_max:
                    new_max = new_e2
                
                if new_max < best_score:
                    # Accept: reverse segment i+1..j
                    cycle[i+1:j+1] = cycle[i+1:j+1][::-1]
                    edge_indices[i] = new_e1
                    edge_indices[j] = new_e2
                    # Recompute internal edges of reversed segment
                    for k in range(i + 1, j):
                        u = cycle[k]
                        v = cycle[(k + 1) % n]
                        edge_indices[k] = _edge_idx(u, v)
                    best_score = new_max
                    improved = True
    
    return cycle, best_score

def _polish(cycle, time_budget):
    """
    High-precision polish: try all single-edge swaps and 2-opt moves
    with a tighter acceptance criterion (accept if new max <= current max).
    This can find equivalent-max solutions that enable further reductions.
    """
    n = len(cycle)
    best_score = _cycle_score(cycle)
    
    # First do standard 2-opt
    cycle, best_score = _two_opt_improve(cycle, time_budget)
    
    # Then try 3-opt moves for additional improvement
    if time.time() < time_budget:
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for i in range(n - 2):
                if time.time() >= time_budget:
                    return cycle, best_score
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n - 1):
                        # 3-opt: try all 3 ways to reconnect 3 cut edges
                        # Edges at i, j, k are cut
                        a, b = cycle[i], cycle[i+1]
                        c, d = cycle[j], cycle[j+1]
                        e, f = cycle[k], cycle[k+1]
                        
                        # Three reconnection options:
                        # 1: (a,c)(b,d)(e,f) -> (a,d)(b,c)(e,f) no that's 2-opt
                        # For 3-opt, we have 3 segments. Try different pairings.
                        # This is complex; skip for now and rely on 2-opt + restarts
                        pass
    
    return cycle, best_score

EDGE_ORDER = [
    [3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23],
    [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19],
    [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19],
    [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8],
    [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17],
    [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23],
    [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4],
    [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12],
    [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20],
    [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16],
    [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18],
    [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21],
    [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14],
    [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6],
    [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8],
    [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18],
    [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12],
    [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23],
    [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20],
    [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5],
    [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11],
    [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7],
    [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9],
    [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4],
    [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14],
    [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23],
    [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16],
    [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]
]

def construct():
    start_time = time.time()
    
    def find_cycle(K, start_node=0, max_time=2.0):
        """
        Finds a Hamilton cycle using only edges from the first K in EDGE_ORDER.
        Returns the list of vertices in the cycle (length N) or None if not found.
        Uses DFS with Warnsdorff's heuristic.
        """
        # Build adjacency bitmasks for the first K edges
        adj = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check min degree >= 2
        for i in range(N):
            if bin(adj[i]).count('1') < 2:
                return None

        visited = [False] * N
        path = [start_node]
        visited[start_node] = True
        
        def dfs(current, depth):
            if time.time() - start_time > max_time:
                return False
            if depth == N:
                if adj[current] & (1 << start_node):
                    return True
                return False
            
            # Warnsdorff's rule: pick neighbor with fewest unvisited neighbors
            candidates = []
            m = adj[current]
            while m:
                bit = m & (-m)
                v = bit.bit_length() - 1
                if not visited[v]:
                    unvisited_count = 0
                    for u in range(N):
                        if not visited[u] and (adj[v] & (1 << u)):
                            unvisited_count += 1
                    candidates.append((unvisited_count, v))
                m ^= bit
            
            if not candidates:
                return False
                
            candidates.sort()
            
            for _, v in candidates:
                visited[v] = True
                path.append(v)
                if dfs(v, depth + 1):
                    return True
                path.pop()
                visited[v] = False
                
            return False

        if dfs(start_node, 1):
            return path
        return None

    # Binary search for minimal K
    lo = N
    hi = N * (N - 1) // 2
    best_cycle = None
    best_K = hi
    
    while lo <= hi and time.time() - start_time < TIME_LIMIT - 10:
        mid = (lo + hi) // 2
        cycle = find_cycle(mid)
        
        if cycle:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if best_cycle:
        # Polish the found cycle with 2-opt
        time_budget = min(time.time() + 15.0, start_time + TIME_LIMIT - 2)
        best_cycle, _ = _two_opt_improve(best_cycle, time_budget)
        return best_cycle
    else:
        return list(range(N))
# EVOLVE-BLOCK-END
