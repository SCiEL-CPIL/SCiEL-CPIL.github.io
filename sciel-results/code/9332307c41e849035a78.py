"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n, start):
    """BFS from start, returns list of distances."""
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in range(n):
            if (adj[u] >> v) & 1 and dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution(adj, n):
    """Check if graph is connected, triangle-free, and has diameter <= 4.
    Returns (is_valid, diameter)."""
    # Check connectivity and diameter
    max_d = 0
    for s in range(n):
        dist = _bfs_distances(adj, n, s)
        if -1 in dist:
            return False, -1
        for d in dist:
            if d > max_d:
                max_d = d
    if max_d > 4:
        return False, max_d
    
    # Check triangle-free
    for u in range(n):
        neighbors_u = adj[u]
        for v in range(u + 1, n):
            if (neighbors_u >> v) & 1:
                # Check if u and v share a common neighbor
                common = adj[u] & adj[v]
                if common != 0:
                    return False, max_d
    
    return True, max_d

def _build_adj(n, edges):
    """Build adjacency bitmask list from edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def construct():
    n = P['n']
    base_edges = P['edges']
    base_adj = _build_adj(n, base_edges)
    
    # Precompute all non-base edges
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (base_adj[i] >> j) & 1:
                all_pairs.append((i, j))
    
    # Precompute: for each pair, does adding it create a triangle with base?
    # Actually we need to check dynamically, but we can precompute which pairs
    # are "safe" in the base graph (no common neighbor in base)
    safe_in_base = []
    for (u, v) in all_pairs:
        common = base_adj[u] & base_adj[v]
        if common == 0:
            safe_in_base.append((u, v))
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Try k = 1, 2, 3, 4, 5, ...
    for k in range(1, 10):
        if time.time() > deadline:
            break
        
        # For k=1: just check each safe edge
        if k == 1:
            for (u, v) in safe_in_base:
                adj = base_adj[:]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                valid, diam = _check_solution(adj, n)
                if valid:
                    return [[int(u), int(v)]]
                if time.time() > deadline:
                    break
            continue
        
        # For k>=2: use combinations with triangle-free pruning
        # We'll use a recursive search with pruning
        def search_combinations(idx, current_edges, current_adj, depth):
            """Recursively build edge set of size k, pruning triangle-free violations."""
            if time.time() > deadline:
                return None
            
            if len(current_edges) == k:
                valid, diam = _check_solution(current_adj, n)
                if valid:
                    return current_edges[:]
                return None
            
            remaining = k - len(current_edges)
            # Need to pick 'remaining' more edges from safe_in_base[idx:]
            if idx + remaining > len(safe_in_base):
                return None
            
            # Try each candidate from idx onwards
            for i in range(idx, len(safe_in_base) - remaining + 1):
                if time.time() > deadline:
                    return None
                
                u, v = safe_in_base[i]
                
                # Check triangle-free with current edges
                common = current_adj[u] & current_adj[v]
                if common != 0:
                    continue
                
                # Add edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append((u, v))
                
                result = search_combinations(i + 1, current_edges, current_adj, depth + 1)
                if result is not None:
                    return result
                
                # Undo
                current_edges.pop()
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            
            return None
        
        result = search_combinations(0, [], base_adj[:], 0)
        if result is not None:
            return [[int(u), int(v)] for u, v in result]
    
    # Fallback: greedy approach if enumeration didn't find small k
    curr_adj = base_adj[:]
    added_edges = []
    
    while time.time() < deadline:
        valid, diam = _check_solution(curr_adj, n)
        if valid:
            break
        
        best_edge = None
        best_diam = float('inf')
        
        for (u, v) in safe_in_base:
            if (curr_adj[u] >> v) & 1:
                continue
            
            # Check triangle-free with current state
            common = curr_adj[u] & curr_adj[v]
            if common != 0:
                continue
            
            curr_adj[u] |= (1 << v)
            curr_adj[v] |= (1 << u)
            v2, d2 = _check_solution(curr_adj, n)
            curr_adj[u] &= ~(1 << v)
            curr_adj[v] &= ~(1 << u)
            
            if not v2:
                continue
            
            if d2 < best_diam:
                best_diam = d2
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        curr_adj[u] |= (1 << v)
        curr_adj[v] |= (1 << u)
        added_edges.append([u, v])
    
    # Prune
    changed = True
    while changed:
        changed = False
        for i in range(len(added_edges) - 1, -1, -1):
            u, v = added_edges[i]
            curr_adj[u] &= ~(1 << v)
            curr_adj[v] &= ~(1 << u)
            valid, _ = _check_solution(curr_adj, n)
            if valid:
                added_edges.pop(i)
                changed = True
            else:
                curr_adj[u] |= (1 << v)
                curr_adj[v] |= (1 << u)
    
    if added_edges:
        final_adj = base_adj[:]
        for u, v in added_edges:
            final_adj[u] |= (1 << v)
            final_adj[v] |= (1 << u)
        valid, _ = _check_solution(final_adj, n)
        if valid:
            return [[int(u), int(v)] for u, v in added_edges]
    
    return []

# EVOLVE-BLOCK-END
