import time
import sys
import numpy as np
from collections import deque
import random

_DEADLINE = 150.0
_T0 = None

def _elapsed():
    return time.time() - _T0

def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def construct(k, g):
    global _T0
    _T0 = time.time()
    
    # Handle small cases with hardcoded cages
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return (n, edges)
    
    if k == 3:
        if g <= 4:
            return (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
        if g <= 5:
            # Petersen
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((5 + i, 5 + (i + 2) % 5))
                edges.append((i, 5 + i))
            return (10, edges)
        if g <= 6:
            # Heawood
            edges = []
            fano_lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
            for li, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7 + li))
            return (14, edges)
        if g <= 7:
            # Tutte-Coxeter
            pass

    # For k=5, g=9, try to find a graph smaller than LUW (1560)
    if k == 5 and g == 9:
        # Try greedy search for n < 1560
        best_result = None
        for n in range(1116, 1561, 20):
            if not _time_ok(margin=10.0):
                break
            result = _greedy_girth_max(n, k, g)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            return best_result
        
        # Fallback to LUW
        return _luw_construct(k, g)

    # General fallback: LUW
    return _luw_construct(k, g)

def _greedy_girth_max(n, k, g):
    """Try to construct a k-regular graph on n vertices with girth >= g using greedy search."""
    if not _time_ok(margin=5.0):
        return None
    
    # Generate random regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
    
    # Initial girth check
    girth = _calculate_girth(adj, n)
    if girth >= g:
        return (n, _adj_to_edges(adj))
    
    # Greedy improvement - find short cycles and break them
    best_girth = girth
    best_adj = [row.copy() for row in adj]
    
    # Limit iterations to avoid infinite loop
    max_iters = 3000
    for it in range(max_iters):
        if not _time_ok(margin=5.0):
            break
        
        # Find a short cycle using BFS
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            break
        
        # Try to break the cycle by swapping an edge in it
        # Pick a random edge from the cycle to remove
        edge_idx = random.randint(0, len(cycle) - 1)
        u, v = cycle[edge_idx]
        
        # Find another edge to swap with: pick random non-adjacent pair
        for attempt in range(50):
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or adj[u2][v2] == 0:
                continue
            if u2 in (u, v) or v2 in (u, v):
                continue
            # Check new edges don't create multi-edges
            if adj[u][v2] != 0 or adj[u2][v] != 0:
                continue
            if u == v2 or u2 == v:
                continue
            
            # Perform swap
            adj[u][v] = 0
            adj[v][u] = 0
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            
            adj[u][v2] = 1
            adj[v2][u] = 1
            adj[u2][v] = 1
            adj[v][u2] = 1
            
            # Check girth
            new_girth = _calculate_girth(adj, n)
            
            if new_girth > best_girth:
                best_girth = new_girth
                best_adj = [row.copy() for row in adj]
                if best_girth >= g:
                    return (n, _adj_to_edges(best_adj))
            else:
                # Revert
                adj[u][v] = 1
                adj[v][u] = 1
                adj[u2][v2] = 1
                adj[v2][u2] = 1
                
                adj[u][v2] = 0
                adj[v2][u] = 0
                adj[u2][v] = 0
                adj[v][u2] = 0
            
            break  # Only try one swap per iteration
    
    if best_girth >= g:
        return (n, _adj_to_edges(best_adj))
    return None

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g using BFS from random starts."""
    for _ in range(20):
        start = random.randint(0, n - 1)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            for v in range(n):
                if adj[u][v] == 1:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and dist[v] <= dist[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = []
                            # Path from u back to start
                            cur = u
                            path1 = []
                            while cur != -1:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.reverse()
                            # Path from v back to start
                            cur = v
                            path2 = []
                            while cur != -1:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.reverse()
                            # Find common ancestor (start)
                            # The cycle is path from u to LCA + path from v to LCA + edge u-v
                            # Simplified: just return the edge (u,v) and some neighbors
                            # For our purposes, we just need one edge on the cycle
                            return [(u, v)]
    return None

def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    # Pair stubs
    adj = np.zeros((n, n), dtype=int)
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if adj[u][v] == 1:
            return None
        adj[u][v] = 1
        adj[v][u] = 1
        
    # Check degrees
    for i in range(n):
        if np.sum(adj[i]) != k:
            return None
            
    return adj

def _calculate_girth(adj, n):
    """Calculate girth using BFS from every vertex."""
    min_girth = float('inf')
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= min_girth:
                continue
            for v in range(n):
                if adj[u][v] == 1:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] > dist[u] + 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            
    return min_girth if min_girth != float('inf') else n + 1

def _adj_to_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
    return edges

def _luw_construct(k, g, d_override=None):
    """Build Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k using vectorized numpy."""
    q = k
    if q not in (4, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        return None

    if d_override is not None:
        d = d_override
    else:
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)

    # Build rule arrays for vectorized computation
    # rule[c] = (a, b) for c in 2..d
    rule_a = np.zeros(d + 1, dtype=np.int64)
    rule_b = np.zeros(d + 1, dtype=np.int64)
    rule_a[2] = 1; rule_b[2] = 1
    rule_a[3] = 2; rule_b[3] = 1
    rule_a[4] = 1; rule_b[4] = 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule_a[c] = 1; rule_b[c] = c0 - 2
        elif r == 1:
            rule_a[c] = c0 - 1; rule_b[c] = 1
        elif r == 2:
            rule_a[c] = c0; rule_b[c] = 1
        else:
            rule_a[c] = 1; rule_b[c] = c0 + 1

    # Field operations
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
        def _add(x, y):
            return x ^ y
        def _sub(x, y):
            return x ^ y
        def _mul(x, y):
            return table[x, y]
    else:
        def _add(x, y):
            return (x + y) % q
        def _sub(x, y):
            return (x - y) % q
        def _mul(x, y):
            return (x * y) % q

    # Enumerate all vertices: (side, x) where side in {0,1}, x is tuple of length d+1 with x[0]=None
    # We'll enumerate side=0 vertices first, then side=1
    # For side=0: x[1..d] range over field elements
    # For side=1: x[1..d] range over field elements
    
    # Total vertices = 2 * q^d
    # For q=5, d=4: 2 * 625 = 1250
    # For q=5, d=5: 2 * 3125 = 6250
    
    # Generate all vertices and compute edges
    # Vertex index: side * q^d + sum(x[i] * q^(d-i) for i in 1..d)
    # x[0] is always 0 (or None)
    
    qd = q ** d
    n_vertices = 2 * qd
    
    # Precompute digit weights
    weights = np.zeros(d, dtype=np.int64)
    for i in range(d):
        weights[i] = q ** (d - 1 - i)
    
    # Generate all x vectors for side=0
    # x is a 1D array of shape (qd, d) where each row is [x[1], ..., x[d]]
    all_x = np.zeros((qd, d), dtype=np.int64)
    for idx in range(qd):
        val = idx
        for i in range(d - 1, -1, -1):
            all_x[idx, i] = val % q
            val //= q
    
    # For each vertex (side, x), compute its neighbors
    # Neighbors of (0, x): for each first in 0..q-1, compute y
    #   y[1] = first
    #   y[c] = x[c] + y[a] * x[b] for side=0
    #   neighbor is (1, y)
    
    # Neighbors of (1, x): for each first in 0..q-1, compute y
    #   y[1] = first
    #   y[c] = x[c] - x[a] * y[b] for side=1
    #   neighbor is (0, y)
    
    # We'll compute all edges by iterating over side=0 vertices and their neighbors
    # Each side=0 vertex has q neighbors on side=1
    
    edges = []
    
    # Process side=0 vertices
    for idx in range(qd):
        if not _time_ok(margin=20.0):
            return None
        x = all_x[idx]
        for first in range(q):
            y = np.zeros(d, dtype=np.int64)
            y[0] = first  # y[1] = first, but our array is 0-indexed for x[1..d]
            for c in range(1, d):
                # y[c+1] = x[c+1] + y[a-1] * x[b-1] (since our arrays are 0-indexed for positions 1..d)
                a = rule_a[c + 1] - 1  # convert to 0-indexed
                b = rule_b[c + 1] - 1
                y[c] = (_add(x[c], _mul(y[a], x[b]))) % q
            # Neighbor is (1, y), index = qd + y[0]*weights[0] + y[1]*weights[1] + ...
            neighbor_idx = qd
            for i in range(d):
                neighbor_idx += y[i] * weights[i]
            # Vertex index for (0, x)
            vert_idx = 0
            for i in range(d):
                vert_idx += x[i] * weights[i]
            edges.append((vert_idx, neighbor_idx))
    
    n = n_vertices
    # Verify
    if _verify(n, edges, k, g):
        return (n, edges)
    return None

def _verify(n, edges, k, g):
    if n == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Non-backtracking walk girth check
    for start in range(n):
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                new_state = (w, v)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if w == start and depth + 1 < g:
                    return False
                queue.append((w, v, depth + 1))
    
    return True
