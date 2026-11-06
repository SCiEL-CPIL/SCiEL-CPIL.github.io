import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

n = 24
edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute arrival index (0-based) for each edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def compute_score(cycle):
    """Compute the score (max 1-based arrival index) of a Hamilton cycle."""
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_arrival(u, v) + 1  # 1-based
        if idx > max_idx:
            max_idx = idx
    return max_idx

def find_hamiltonian_cycle(k, time_limit):
    """
    Find a Hamiltonian cycle in the subgraph formed by the first k edges.
    Uses Warnsdorff's heuristic and connectivity pruning.
    Returns the cycle as a list of vertices if found, None otherwise.
    """
    # Build adjacency list for G_k
    adj = [[] for _ in range(n)]
    for i in range(k):
        u, v = edge_order[i]
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree constraints - every vertex must have degree >= 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None
    
    deadline = time.time() + time_limit
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    def is_connected_unvisited():
        """Check that all unvisited vertices are connected (plus the endpoints of the path)."""
        # Collect unvisited vertices
        unvisited = [u for u in range(n) if not visited[u]]
        if not unvisited:
            return True
        
        # BFS from the first unvisited vertex
        start = unvisited[0]
        visited_bfs = [False] * n
        queue = deque([start])
        visited_bfs[start] = True
        count = 1
        
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if not visited_bfs[nb] and not visited[nb]:
                    visited_bfs[nb] = True
                    count += 1
                    queue.append(nb)
        
        return count == len(unvisited)
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            last = path[-1]
            # Check if last connects to start (vertex 0)
            if 0 in adj[last]:
                return path[:]
            return None
        
        current = path[-1]
        
        # Get unvisited neighbors
        candidates = []
        for nb in adj[current]:
            if not visited[nb]:
                # Warnsdorff's heuristic: count unvisited neighbors of nb
                # (excluding current, which is already visited)
                unvisited_count = 0
                for nnb in adj[nb]:
                    if not visited[nnb]:
                        unvisited_count += 1
                candidates.append((unvisited_count, nb))
        
        if not candidates:
            return None
        
        # Sort by Warnsdorff: fewest onward options first
        candidates.sort()
        
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            
            # Prune: check connectivity of remaining unvisited vertices
            if len(path) < n - 1:  # Don't check at the very end
                if not is_connected_unvisited():
                    path.pop()
                    visited[nb] = False
                    continue
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            visited[nb] = False
            
        return None
        
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Default cycle
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Binary search for the minimum k where a Hamiltonian cycle exists
    # Lower bound: n edges (24), Upper bound: 276
    lo = 24
    hi = 276
    
    # First, verify that a cycle exists at hi
    if time.time() > deadline - 10:
        return best_cycle
    
    # Check if cycle exists at hi with a generous time limit
    cycle_hi = find_hamiltonian_cycle(hi, 20)
    if cycle_hi is not None:
        score_hi = compute_score(cycle_hi)
        if score_hi < best_score:
            best_score = score_hi
            best_cycle = cycle_hi
        hi = score_hi  # Tighten upper bound
    
    # Binary search
    while lo < hi and time.time() < deadline - 15:
        mid = (lo + hi) // 2
        t_limit = max(0.5, min(5.0, (deadline - time.time()) / 3 - 15))
        
        cycle = find_hamiltonian_cycle(mid, t_limit)
        if cycle is not None:
            score = compute_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    # Final check at lo
    if time.time() < deadline - 5:
        cycle = find_hamiltonian_cycle(lo, 10)
        if cycle is not None:
            score = compute_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
    
    return best_cycle

# EVOLVE-BLOCK-END
