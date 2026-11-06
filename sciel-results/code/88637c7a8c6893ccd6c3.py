import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Precompute edge indices (1-based)
_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def _greedy_cycle():
    """Greedy heuristic: always pick the neighbor with the smallest edge index."""
    visited = [False] * N
    visited[0] = True
    current = 0
    cycle = [0]
    
    for _ in range(N - 1):
        best_next = -1
        best_idx = float('inf')
        for j in range(N):
            if not visited[j]:
                idx = _edge_idx(current, j)
                if idx < best_idx:
                    best_idx = idx
                    best_next = j
        if best_next == -1:
            return None
        visited[best_next] = True
        cycle.append(best_next)
        current = best_next
        
    # Check closing edge
    if _edge_idx(0, current) < float('inf'):
        return cycle
    return None

def _improve_cycle(cycle, max_k, time_limit=0.1):
    """Local search: 2-opt swaps to reduce the maximum edge index."""
    current_cycle = list(cycle)
    current_score = _cycle_score(current_cycle)
    if current_score > max_k:
        return current_cycle, current_score
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                if i == 0 and j == N - 1:
                    continue
                # 2-opt move: reverse segment between i and j
                # Original edges: (c[i-1], c[i]) and (c[j], c[j+1]) [mod N]
                # New edges: (c[i-1], c[j]) and (c[i], c[j+1])
                a = current_cycle[i - 1]
                b = current_cycle[i]
                c = current_cycle[j]
                d = current_cycle[(j + 1) % N]
                
                # Check if the new edges are within max_k
                if _edge_idx(a, c) > max_k or _edge_idx(b, d) > max_k:
                    continue
                    
                # Perform swap
                new_cycle = current_cycle[:i] + current_cycle[i:j+1][::-1] + current_cycle[j+1:]
                new_score = _cycle_score(new_cycle)
                
                if new_score < current_score:
                    current_cycle = new_cycle
                    current_score = new_score
                    improved = True
    return current_cycle, current_score

def _dfs_find_cycle(adj, K, deadline):
    """
    Depth-first search to find a Hamilton cycle using only edges with index <= K.
    Uses Warnsdorff's rule (minimum degree heuristic) for ordering.
    Returns a list representing the cycle if found, else None.
    """
    n = N
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    # Precompute degrees for Warnsdorff
    # Actually, dynamic degree is better, but let's stick to simple heuristic first.
    # Heuristic: sort neighbors by number of unvisited neighbors (smallest first)
    
    def dfs(current):
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check if we can close the loop
            if _edge_idx(current, 0) <= K:
                return True
            return False
            
        # Get unvisited neighbors
        neighbors = []
        for v in adj[current]:
            if not visited[v]:
                neighbors.append(v)
                
        if not neighbors:
            return False
            
        # Warnsdorff: sort by number of unvisited neighbors of the neighbor
        # This is O(N^2) per step, but N=24 is small.
        def count_unvisited(v):
            cnt = 0
            for w in adj[v]:
                if not visited[w]:
                    cnt += 1
            return cnt
            
        neighbors.sort(key=count_unvisited)
        
        for next_node in neighbors:
            visited[next_node] = True
            path.append(next_node)
            if dfs(next_node):
                return True
            path.pop()
            visited[next_node] = False
            
        return False
        
    if dfs(0):
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    # 1. Get initial upper bound
    greedy_res = _greedy_cycle()
    if greedy_res is None:
        # Fallback: try a simple permutation
        return list(range(N))
        
    best_cycle = greedy_res
    best_K = _cycle_score(best_cycle)
    
    # 2. Improve the initial cycle using local search
    best_cycle, best_K = _improve_cycle(best_cycle, best_K, time_limit=1.0)
    
    # 3. Binary search for minimal K using DFS
    # Lower bound: A Hamilton cycle needs at least N edges.
    # The N-th smallest edge index in the complete graph is N (since indices are 1..M).
    # But actually, the minimum possible score is at least the N-th smallest edge in the *cycle*.
    # A safe lower bound is 1. But we can be smarter.
    # The minimum degree of any vertex in the prefix graph must be at least 2.
    # Let's just start from 1.
    
    lo = 1
    hi = best_K
    
    # We will search for the smallest K such that a cycle exists.
    # We use the best_K found so far as the upper bound.
    
    last_feasible_cycle = best_cycle
    last_feasible_K = best_K
    
    # We can also try to improve the cycle found at a specific K.
    
    while lo < hi:
        if time.time() > deadline - 5:
            break
            
        mid = (lo + hi) // 2
        
        # Build adjacency list for edges with index <= mid
        adj = [[] for _ in range(N)]
        for u in range(N):
            for v in range(N):
                if u == v:
                    continue
                if _edge_idx(u, v) <= mid:
                    adj[u].append(v)
                    
        # Check if graph is connected and min degree >= 2
        # Quick check: if any vertex has degree < 2, skip
        feasible = True
        for u in range(N):
            if len(adj[u]) < 2:
                feasible = False
                break
                
        if feasible:
            cycle = _dfs_find_cycle(adj, mid, deadline - 5)
            if cycle is not None:
                # Found a cycle with max edge <= mid
                # Try to improve it locally
                cycle, score = _improve_cycle(cycle, mid, time_limit=0.5)
                if score < last_feasible_K:
                    last_feasible_K = score
                    last_feasible_cycle = cycle
                hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1
            
    # Final verification
    final_score = _cycle_score(last_feasible_cycle)
    
    return last_feasible_cycle
# EVOLVE-BLOCK-END
