"""Find a Hamilton cycle with minimized max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import random

_N = 24
_DEADLINE = 85.0  # seconds margin before 90s limit

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

def _has_hamilton_cycle_in_prefix(k):
    """Check if a Hamilton cycle exists using only edges with arrival index <= k.
    Uses DFS with backtracking and aggressive pruning."""
    # Build adjacency: for each vertex, neighbors reachable with edge index <= k
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i][j] <= k:
                adj[i].append(j)
                adj[j].append(i)
    
    # Check minimum degree: every vertex must have degree >= 2
    for i in range(_N):
        if len(adj[i]) < 2:
            return False
    
    # Fix vertex 0 as start (symmetry breaking)
    FULL = (1 << _N) - 1
    
    def dfs(current, visited, path_len):
        if path_len == _N:
            # Check if we can close the cycle: edge from current to 0
            if _edge_arrival[current][0] <= k:
                return True
            return False
        
        # Pruning: check that all unvisited vertices have at least one neighbor
        # that is either unvisited or is the current vertex (for potential continuation)
        remaining = FULL & ~visited
        rem_count = bin(remaining).count('1')
        
        for nxt in adj[current]:
            if visited & (1 << nxt):
                continue
            if path_len == _N - 1:
                # Last vertex: must connect back to 0
                if _edge_arrival[nxt][0] <= k:
                    return True
                continue
            
            # Pruning: after visiting nxt, check that unvisited vertices still have enough connections
            new_visited = visited | (1 << nxt)
            rem = FULL & ~new_visited
            
            # Check all unvisited vertices have at least one unvisited neighbor or neighbor in {current, nxt}
            valid = True
            r = rem
            while r:
                v = (r & -r).bit_length() - 1
                r &= r - 1
                if v == 0:
                    continue
                # v must have at least one neighbor that is unvisited (not yet visited)
                has_unvisited_neighbor = False
                for nn in adj[v]:
                    if not (new_visited & (1 << nn)):
                        has_unvisited_neighbor = True
                        break
                if not has_unvisited_neighbor:
                    valid = False
                    break
            if not valid:
                continue
            
            if dfs(nxt, new_visited, path_len + 1):
                return True
        
        return False
    
    return dfs(0, 1, 1)

def _find_cycle_in_prefix(k):
    """Find a Hamilton cycle using only edges with arrival index <= k. Returns cycle or None."""
    adj = [[] for _ in range(_N)]
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i][j] <= k:
                adj[i].append(j)
                adj[j].append(i)
    
    FULL = (1 << _N) - 1
    path = []
    
    def dfs(current, visited, path_len):
        path.append(current)
        if path_len == _N:
            if _edge_arrival[current][0] <= k:
                return True
            path.pop()
            return False
        
        for nxt in adj[current]:
            if visited & (1 << nxt):
                continue
            if path_len == _N - 1:
                if _edge_arrival[nxt][0] <= k:
                    path.append(nxt)
                    return True
                continue
            
            # Pruning
            new_visited = visited | (1 << nxt)
            rem = FULL & ~new_visited
            if rem:
                # Check all unvisited non-zero vertices have at least one edge to unvisited or to path
                valid = True
                r = rem
                while r:
                    v = (r & -r).bit_length() - 1
                    r &= r - 1
                    if v == 0:
                        continue
                    # v must have at least one neighbor in unvisited (excluding self) or in current path end
                    cnt = 0
                    for nn in adj[v]:
                        if (new_visited & (1 << nn)) == 0:
                            cnt += 1
                            if cnt >= 1:
                                break
                    if cnt < 1:
                        valid = False
                        break
                if not valid:
                    continue
            
            if dfs(nxt, new_visited, path_len + 1):
                return True
        
        path.pop()
        return False
    
    if dfs(0, 1, 1):
        return path[:]
    return None

def _two_opt_improve(cycle, max_k, rng, time_limit):
    """Improve a cycle by 2-opt moves, trying to keep max edge index <= max_k.
    Returns improved cycle and its score."""
    n = len(cycle)
    best_cycle = cycle[:]
    best_score = _cycle_score(best_cycle)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try all 2-opt moves
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment i+1..j
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    cycle = new_cycle
                    improved = True
                    if time.time() > time_limit:
                        break
            if time.time() > time_limit:
                break
    
    return best_cycle, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # First, get a baseline: the identity cycle
    baseline = list(range(_N))
    best_cycle = baseline
    best_score = _cycle_score(baseline)
    
    # Binary search for the minimum k where a Hamilton cycle exists
    # The edge indices range from 1 to C(24,2) = 276
    lo, hi = 1, 276
    
    # First find the minimum k via binary search
    found_k = None
    while lo <= hi and time.time() < deadline - 10.0:
        mid = (lo + hi) // 2
        if _has_hamilton_cycle_in_prefix(mid):
            found_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if found_k is not None:
        # Now find an actual cycle at k = found_k
        cycle = None
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 20:
            cycle = _find_cycle_in_prefix(found_k)
            if cycle is not None:
                break
            attempts += 1
        
        if cycle is not None:
            score = _cycle_score(cycle)
            if score < best_score:
                best_cycle = cycle
                best_score = score
    
    # If binary search didn't find anything in time, try greedy/random approaches
    if best_score >= 200:  # if we didn't make progress
        # Try random restarts with local search
        rng = random.Random(42)
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            # Random permutation
            perm = list(range(_N))
            rng.shuffle(perm)
            # 2-opt improvement: minimize max edge
            perm, score = _two_opt_improve(perm, best_score, rng, deadline - 5.0)
            if score < best_score:
                best_score = score
                best_cycle = perm
    
    return best_cycle

# EVOLVE-BLOCK-END
