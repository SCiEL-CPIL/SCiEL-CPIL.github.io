import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

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
        idx = get_arrival(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1  # 1-based

def find_hamiltonian_cycle(max_arrival, time_limit, start_vertex=0):
    """
    Find a Hamiltonian cycle using only edges with arrival <= max_arrival.
    Uses Warnsdorff's heuristic: prefer neighbors with fewer unvisited options.
    Returns the cycle as a list of vertices, or None if not found.
    """
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for idx in range(max_arrival + 1):
        u, v = edge_order[idx]
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree constraint: every vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Pre-sort neighbors by arrival time as tiebreaker
    for v in range(n):
        adj[v].sort(key=lambda x: get_arrival(v, x))
    
    visited = [False] * n
    path = [start_vertex]
    visited[start_vertex] = True
    
    deadline = time.time() + time_limit
    result = [None]
    
    def backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check if we can close the cycle
            last = path[-1]
            if get_arrival(last, start_vertex) <= max_arrival:
                result[0] = path[:]
                return True
            return False
        
        current = path[-1]
        remaining = n - len(path)
        
        # Get unvisited neighbors
        candidates = []
        for nb in adj[current]:
            if visited[nb]:
                continue
            # Pruning: if only one vertex left to visit, it must connect to start
            if remaining == 2:
                if get_arrival(nb, start_vertex) > max_arrival:
                    continue
            candidates.append(nb)
        
        if not candidates:
            return False
        
        # Warnsdorff's heuristic: sort by number of unvisited neighbors (fewest first)
        # Tiebreak by arrival time
        def warnsdorff_key(nb):
            unvisited_count = sum(1 for x in adj[nb] if not visited[x])
            return (unvisited_count, get_arrival(current, nb))
        
        candidates.sort(key=warnsdorff_key)
        
        for nb in candidates:
            visited[nb] = True
            path.append(nb)
            
            if backtrack():
                return True
            
            path.pop()
            visited[nb] = False
        
        return False
    
    backtrack()
    return result[0]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Default cycle: 0, 1, 2, ..., 23, 0
    default_cycle = list(range(n))
    best_score = compute_score(default_cycle)
    best_cycle = default_cycle
    
    # Binary search for the minimal threshold where a Hamiltonian cycle exists
    # First, establish an upper bound
    lo, hi = 23, 275  # 0-based indices
    
    # Find upper bound: try increasing thresholds
    upper = hi
    for mid in range(23, 276, 5):
        if time.time() > deadline - 10:
            break
        cycle = find_hamiltonian_cycle(mid, 0.5)
        if cycle is not None:
            upper = mid
            break
    else:
        # If we didn't find one in the scan, use the last tried
        upper = 275
    
    # Binary search between lo and upper
    while lo < upper and time.time() < deadline - 15:
        mid = (lo + upper) // 2
        time_per_attempt = min(2.0, max(0.1, (deadline - 15 - time.time()) * 0.15))
        cycle = find_hamiltonian_cycle(mid, time_per_attempt)
        if cycle is not None:
            upper = mid
            score = compute_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
        else:
            lo = mid + 1
    
    # Final polish: try to find a cycle at the best threshold with more time
    if time.time() < deadline - 5:
        final_threshold = upper
        time_left = deadline - 5 - time.time()
        # Try multiple starting vertices
        for sv in range(n):
            if time.time() > deadline - 5:
                break
            cycle = find_hamiltonian_cycle(final_threshold, time_left / (n - sv), start_vertex=sv)
            if cycle is not None:
                score = compute_score(cycle)
                if score < best_score:
                    best_score = score
                    best_cycle = cycle
                break  # Found a valid cycle at this threshold
    
    return best_cycle
# EVOLVE-BLOCK-END
