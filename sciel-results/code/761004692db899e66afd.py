import time
import random
from collections import defaultdict

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

# Precompute lower bound: max index of the 2nd edge incident to any vertex
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
        
    # We need the max of the indices where degree became 2
    # If a vertex never reaches degree 2, it's invalid, but complete graph ensures it does.
    max_idx = 0
    for v in range(N):
        if last_edge_idx[v] > max_idx:
            max_idx = last_edge_idx[v]
    return max_idx

MIN_K_LOWER_BOUND = _compute_lower_bound()

# Also compute a tighter lower bound: for each vertex, find the index of its 2nd edge
# The Hamilton cycle must include exactly 2 edges per vertex, so the 2nd edge of every
# vertex must be present.
def _tighter_lower_bound():
    """For each vertex, find the arrival index of its 2nd incident edge.
    The answer must be >= max over all vertices of this index."""
    second_edge_idx = [0] * N
    for v in range(N):
        count = 0
        for i, (u, w) in enumerate(EDGE_ORDER):
            idx = i + 1
            if u == v or w == v:
                count += 1
                if count == 2:
                    second_edge_idx[v] = idx
                    break
    return max(second_edge_idx)

TIGHTER_LB = _tighter_lower_bound()
MIN_K_LOWER_BOUND = max(MIN_K_LOWER_BOUND, TIGHTER_LB)

def _build_adjacency(K):
    """Build adjacency list for edges with arrival index <= K."""
    adj = [set() for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER):
        if i + 1 > K:
            break
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _warnsdorff_dfs(adj, start, max_nodes=5000000):
    """DFS with Warnsdorff's heuristic."""
    n = N
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    def dfs(path):
        if len(path) == n:
            # Check if closing edge exists
            if path[-1] in adj[path[0]]:
                return True
            return False
        
        current = path[-1]
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return False
        
        # Warnsdorff's rule: visit neighbor with fewest unvisited neighbors first
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

def _randomized_greedy_hamilton_cycle(adj, start, max_restarts=100):
    """Try to find a Hamilton cycle using randomized greedy approach."""
    n = N
    for _ in range(max_restarts):
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        success = True
        for _ in range(n - 1):
            current = path[-1]
            neighbors = [v for v in adj[current] if not visited[v]]
            if not neighbors:
                success = False
                break
            
            # Sort by degree, then pick randomly from top few
            neighbors.sort(key=lambda v: len([w for w in adj[v] if not visited[w]]))
            
            # Pick from top 3 or all if fewer
            k = min(3, len(neighbors))
            choice = random.choice(neighbors[:k])
            
            visited[choice] = True
            path.append(choice)
        
        if success and path[-1] in adj[path[0]]:
            return path
    return None

def _find_hamilton_cycle(K, time_limit=2.0):
    """Try to find a Hamilton cycle in the graph with first K edges."""
    adj = _build_adjacency(K)
    
    # Quick check: every vertex must have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    start_time = time.time()
    
    # Try Warnsdorff from each starting vertex
    for s in range(N):
        if time.time() - start_time > time_limit:
            break
        cycle = _warnsdorff_dfs(adj, s)
        if cycle is not None:
            return cycle
            
    # If Warnsdorff fails, try randomized greedy
    for s in range(N):
        if time.time() - start_time > time_limit:
            break
        cycle = _randomized_greedy_hamilton_cycle(adj, s, max_restarts=50)
        if cycle is not None:
            return cycle
            
    return None

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
    """Try to improve a cycle's score using 2-opt, Or-opt, and swap moves."""
    n = len(cycle)
    best_score = _cycle_score(cycle)
    
    # Precompute edge positions for fast lookup
    pos = {}
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        key = _edge_key(u, v)
        pos[key] = edge_pos.get(key, float('inf'))
    
    def get_score(cyc):
        m = 0
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            idx = edge_pos.get(_edge_key(u, v), float('inf'))
            if idx > m: m = idx
        return m

    def try_move(new_cycle):
        nonlocal cycle, best_score
        s = get_score(new_cycle)
        if s < best_score:
            cycle = new_cycle
            best_score = s
            return True
        return False

    for _ in range(max_iter):
        improved = False
        
        # 2-Opt: reverse segment
        for i in range(n):
            for j in range(i + 2, n):
                if (i == 0 and j == n - 1):
                    continue
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                if try_move(new_cycle):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Or-Opt: move single vertex
        for i in range(1, n):
            v = cycle[i]
            new_cycle = cycle[:i] + cycle[i+1:]
            for j in range(len(new_cycle)):
                candidate = new_cycle[:j] + [v] + new_cycle[j:]
                if try_move(candidate):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Swap two vertices (valid cycle move)
        for i in range(1, n):
            for j in range(i+1, n):
                new_cycle = cycle[:]
                new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
                if try_move(new_cycle):
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        if not improved:
            break
    return cycle, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s margin
    
    # Binary search on K
    lo = MIN_K_LOWER_BOUND
    hi = 276
    best_cycle = None
    best_K = hi
    
    # First, try to find any cycle at full K (should always work)
    cycle = _find_hamilton_cycle(hi)
    if cycle is not None:
        best_cycle = cycle
        best_K = hi
    
    # Binary search for minimum K
    while lo < hi and (time.time() - start_time) < deadline - 15:
        mid = (lo + hi) // 2
        cycle = _find_hamilton_cycle(mid, time_limit=2.0)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Refinement: try to improve the best cycle found with extended local search
    if best_cycle is not None:
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            # Use more iterations for local search
            max_iter = min(50000, int(remaining * 5000))
            improved_cycle, improved_score = _two_opt_improve(best_cycle, _edge_pos, max_iter=max_iter)
            if improved_score < _cycle_score(best_cycle):
                best_cycle = improved_cycle
    
    # Final check
    if best_cycle is None:
        # Fallback: identity permutation
        return list(range(N))
    
    return best_cycle
# EVOLVE-BLOCK-END
