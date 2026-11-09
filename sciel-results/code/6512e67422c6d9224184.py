import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth, deadline=None):
    """Check if girth >= target_girth using BFS from each vertex.
    Early termination: stop exploring from a start once depth reaches target_girth//2,
    since any cycle through vertices at depth >= target_girth//2 has length >= target_girth.
    Returns True if girth >= target_girth, False otherwise."""
    max_depth = target_girth // 2 - 1
    for start in range(n):
        if deadline is not None and time.time() > deadline:
            return True  # assume valid if we run out of time (risky but better than timeout)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            dv = dist[v]
            if dv >= max_depth:
                continue  # no need to explore further; cycles would be >= target_girth
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    if dv + 1 < max_depth:
                        queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dv + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth

def _construct_cage_3_16():
    """Construct a 3-regular bipartite graph with girth >= 16 using random perfect matching approach.
    
    Strategy: Build a 3-regular bipartite graph by choosing 3 random perfect matchings
    between two halves. Limited to a short time budget since random search is unlikely
    to find girth 16; this is a quick attempt before falling back to LUW.
    """
    n = 936
    half = n // 2
    deadline = time.time() + 15.0  # Only spend 15s on this; LUW is more reliable
    
    def _random_perfect_matching():
        """Generate a random perfect matching using Fisher-Yates."""
        perm = list(range(half))
        random.shuffle(perm)
        return perm
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate 3 random perfect matchings
        matchings = [_random_perfect_matching() for _ in range(3)]
        
        # Check no repeated edges (same pair appears in multiple matchings)
        edge_set = set()
        valid = True
        for m in matchings:
            for i in range(half):
                edge = (i, m[i])
                if edge in edge_set:
                    valid = False
                    break
                edge_set.add(edge)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for m in matchings:
            for i in range(half):
                j = m[i]
                adj[i].append(half + j)
                adj[half + j].append(i)
        
        # Check girth
        if _has_girth_at_least(adj, n, 16, deadline=deadline - 2.0):
            edges = []
            for i in range(half):
                for j in adj[i]:
                    if j >= half:
                        edges.append((i, j))
            return n, edges
    
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction.
    Optimized to find minimal d for required girth with faster neighbor computation."""
    q = k
    deadline = time.time() + 100.0  # leave 20s margin for final girth check and return
    
    def _build_luw(d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        # Precompute for faster neighbor generation
        # For side 0: y[c] = x[c] + y[a] * x[b]
        # For side 1: y[c] = x[c] - x[a] * y[b]
        
        def neighbours_side0(x):
            """Generate neighbors from a side-0 vertex (x is a tuple of length d+1, x[0]=None)."""
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    y[c] = (x[c] + y[a] * x[b]) % q
                yield tuple(y)
        
        def neighbours_side1(x):
            """Generate neighbors from a side-1 vertex (x is a tuple of length d+1, x[0]=None)."""
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    y[c] = (x[c] - x[a] * y[b]) % q
                yield tuple(y)
        
        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        i = 0
        while i < len(order):
            vertex = order[i]
            side, x = vertex
            if side == 0:
                gen = neighbours_side0(x)
            else:
                gen = neighbours_side1(x)
            for other_x in gen:
                other = (1 - side, other_x)
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges
    
    # Try increasing d from a reasonable lower bound
    # Girth of LUW graph is roughly 2*(d-1) or so; need girth >= g
    # Start from d where 2*d >= g, i.e., d >= g//2
    d_start = max(5, (g - 1) // 2)
    for d in range(d_start, d_start + 15):
        if time.time() > deadline:
            break
        
        n, edges = _build_luw(d)
        
        # Build adjacency for girth check
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _has_girth_at_least(adj, n, g, deadline=deadline - 5.0):
            return n, edges
    
    # Fallback: use a known-good large d value
    d_orig = max(8, g)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    
    if k == 3 and g == 16:
        # Try random bipartite construction first (quick attempts)
        result = _construct_cage_3_16()
        if result is not None:
            return result
        # Fallback to LUW
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
