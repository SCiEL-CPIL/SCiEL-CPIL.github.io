import time
import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

# Precompute edge positions
edge_pos = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[frozenset((u, v))] = idx

# Calculate constant C based on identity cycle score = 1.0
identity_cycle = list(range(N))
max_idx_id = 0
for i in range(N):
    u = identity_cycle[i]
    v = identity_cycle[(i + 1) % N]
    e = frozenset((u, v))
    pos = edge_pos[e]
    if pos > max_idx_id:
        max_idx_id = pos
C = max_idx_id

# Build adjacency lists sorted by arrival index
adj = [[] for _ in range(N)]
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    adj[u].append((v, idx))
    adj[v].append((u, idx))
for v in range(N):
    adj[v].sort(key=lambda x: x[1])

def has_hamilton_cycle(K, time_limit=5.0):
    """
    Check if a Hamilton cycle exists in the graph formed by edges with index <= K.
    Uses backtracking with pruning.
    """
    start_time = time.time()
    
    # Build adjacency for G_K
    adj_k = [[] for _ in range(N)]
    for v in range(N):
        for (nbr, pos) in adj[v]:
            if pos <= K:
                adj_k[v].append(nbr)
            else:
                break
    
    # Basic checks
    for v in range(N):
        if len(adj_k[v]) < 2:
            return None
            
    # Connectivity check
    visited = [False] * N
    visited[0] = True
    stack = [0]
    count = 1
    while stack:
        u = stack.pop()
        for w in adj_k[u]:
            if not visited[w]:
                visited[w] = True
                count += 1
                stack.append(w)
    if count != N:
        return None

    # Backtracking
    path = [0]
    used = [False] * N
    used[0] = True
    
    # Precompute neighbors sets for faster lookup? 
    # adj_k is list of lists.
    
    def backtrack():
        if time.time() - start_time > time_limit:
            return False
            
        if len(path) == N:
            last = path[-1]
            # Check if last connects to 0
            if 0 in adj_k[last]:
                return True
            return False
        
        cur = path[-1]
        
        # Heuristic: Sort neighbors by degree in remaining graph (ascending)
        # This is "Minimum Remaining Values"
        # We want to pick the neighbor that leaves the fewest options for the next step
        # or rather, the neighbor that is most constrained.
        
        candidates = []
        for w in adj_k[cur]:
            if not used[w]:
                # Calculate degree of w in the subgraph of unvisited nodes + cur
                # This is expensive if done naively.
                # Approximation: just use the static degree in adj_k[w]
                # Or better: count how many unvisited neighbors w has
                deg = 0
                for x in adj_k[w]:
                    if not used[x] or x == cur:
                        deg += 1
                candidates.append((deg, w))
                
        # Sort by degree (ascending) - most constrained first
        candidates.sort()
        
        for deg, w in candidates:
            if used[w]:
                continue
                
            # Pruning: If w has no other unvisited neighbors (except cur), 
            # it must be the last node in the path (before closing to 0).
            # If it's not the last node, it's a dead end.
            if deg == 1 and len(path) < N - 1:
                # w only connects to cur in the remaining graph.
                # So if we go to w, we must stop there (if it's the last) 
                # or we are stuck.
                # If len(path) < N-1, we need to continue from w to another unvisited node.
                # But w has no other unvisited neighbors. So this is a dead end.
                continue

            used[w] = True
            path.append(w)
            if backtrack():
                return True
            path.pop()
            used[w] = False
            
        return False

    if backtrack():
        return path
    return None

def _find_best_cycle():
    deadline = time.time() + 80  # Safety margin
    
    lo, hi = 24, M
    best_K = M
    best_cycle = None
    
    # Binary search for minimum K
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = has_hamilton_cycle(mid, time_limit=10.0)
        if cycle is not None:
            best_K = mid
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback: Greedy
        # Just return identity if we fail completely
        return identity_cycle
        
    return best_cycle

_cached_result = None

def construct():
    global _cached_result
    if _cached_result is not None:
        return _cached_result
    
    cycle = _find_best_cycle()
    _cached_result = cycle
    return cycle

# EVOLVE-BLOCK-END
