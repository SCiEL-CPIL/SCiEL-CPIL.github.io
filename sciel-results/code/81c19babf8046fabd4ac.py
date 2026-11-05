"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build edge index lookup
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    if u > v:
        u, v = v, u
    edge_idx[(u, v)] = i

def edge_weight(u, v):
    if u > v:
        u, v = v, u
    return edge_idx[(u, v)]

def build_adj_for_k(k):
    """Build adjacency list for subgraph G_k (edges with arrival index <= k)."""
    adj = [set() for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > k:
            break
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_hamilton_cycle_dfs(adj, start, deadline):
    """
    Check if G has a Hamilton cycle using DFS with pruning.
    Returns the cycle if found, None otherwise.
    Uses degree-based and connectivity pruning.
    """
    n = N
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    # Precompute degrees
    deg = [len(adj[i]) for i in range(n)]
    
    def dfs(current, depth):
        if time.time() > deadline:
            return False
        
        if depth == n - 1:
            # Check if closing edge exists
            if start in adj[current]:
                return True
            return False
        
        # Pruning: check if remaining unvisited vertices can still form a path
        unvisited = [v for v in range(n) if not visited[v]]
        
        # Degree check: any unvisited vertex with degree < 2 in remaining graph is a problem
        # (it needs at least 2 edges in the final cycle, but one might be to a visited vertex)
        for v in unvisited:
            # Count neighbors that are either unvisited or are start (for closing edge)
            avail = 0
            for nb in adj[v]:
                if not visited[nb] or nb == start:
                    avail += 1
            if avail < 2:
                return False
        
        # Try neighbors in order of increasing degree (most constrained first)
        neighbors = [v for v in adj[current] if not visited[v]]
        neighbors.sort(key=lambda v: deg[v])
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            visited[v] = False
            
            # Additional pruning: if v is now isolated in remaining graph
            # (only connected to visited nodes and not start)
            if depth < n - 2:
                remaining_unvisited = [x for x in range(n) if not visited[x] and x != v]
                for x in remaining_unvisited:
                    avail = 0
                    for nb in adj[x]:
                        if not visited[nb] or nb == start or nb == v:
                            avail += 1
                    if avail < 2:
                        break
                # This check is expensive; only do it sometimes
                # Actually, let's skip this for speed and rely on the earlier check
        
        return False
    
    return dfs(start, 0)

def find_cycle_for_k(k, deadline, rng=None):
    """Find a Hamilton cycle in G_k. Returns cycle or None."""
    adj = build_adj_for_k(k)
    
    # Check if all vertices have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Try multiple starting points with randomization
    starts = list(range(N))
    if rng:
        rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            return None
        
        # Reset visited
        visited = [False] * N
        path = [start]
        visited[start] = True
        deg = [len(adj[i]) for i in range(N)]
        
        def dfs(current, depth):
            if time.time() > deadline:
                return False
            
            if depth == N - 1:
                if start in adj[current]:
                    return True
                return False
            
            # Pruning: degree check for unvisited vertices
            for v in range(N):
                if not visited[v]:
                    avail = 0
                    for nb in adj[v]:
                        if not visited[nb] or nb == start:
                            avail += 1
                    if avail < 2:
                        return False
            
            neighbors = [v for v in adj[current] if not visited[v]]
            # Sort by degree (most constrained first)
            neighbors.sort(key=lambda v: deg[v])
            
            for v in neighbors:
                visited[v] = True
                path.append(v)
                if dfs(v, depth + 1):
                    return True
                path.pop()
                visited[v] = False
            
            return False
        
        if dfs(start, 0):
            return path[:]
    
    return None

def two_opt_improve(cycle, max_iters=200):
    """Quick 2-opt to improve the cycle found."""
    n = len(cycle)
    best = cycle[:]
    best_score = max(edge_weight(best[i], best[(i+1) % n]) for i in range(n))
    
    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_score = max(edge_weight(new_cycle[k], new_cycle[(k+1) % n]) for k in range(n))
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
        if not improved:
            break
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Binary search on K
    # Lower bound: minimum K such that all vertices have degree >= 2 in G_K
    lo = 1
    hi = len(EDGE_ORDER)  # 276
    
    # Find a feasible upper bound
    best_cycle = None
    best_k = hi
    
    # First, find a reasonable upper bound by trying some K values
    # The minimum possible K is at least N (need at least N edges for a cycle)
    lo = max(lo, N)
    
    # Binary search
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        
        if time.time() > deadline - 3:
            break
        
        cycle = find_cycle_for_k(mid, deadline - 3)
        
        if cycle is not None:
            best_cycle = cycle
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a cycle, try to improve it with 2-opt
    if best_cycle is not None:
        if time.time() < deadline - 2:
            best_cycle, _ = two_opt_improve(best_cycle, max_iters=100)
    
    # Fallback: if no cycle found via binary search, use greedy
    if best_cycle is None:
        if time.time() < deadline - 2:
            # Simple greedy fallback
            visited = [False] * N
            cycle = [0]
            visited[0] = True
            current = 0
            for _ in range(N - 1):
                best_next = -1
                best_w = float('inf')
                for v in range(N):
                    if not visited[v]:
                        w = edge_weight(current, v)
                        if w < best_w:
                            best_w = w
                            best_next = v
                cycle.append(best_next)
                visited[best_next] = True
                current = best_next
            best_cycle = cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
