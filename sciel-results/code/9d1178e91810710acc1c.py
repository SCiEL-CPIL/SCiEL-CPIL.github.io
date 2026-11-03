import time
import sys
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    if k != 3:
        # Generic fallback for other k
        return _construct_luw(k, g)

    best_n = float('inf')
    best_edges = None

    # Moore bound for (3, 16): 2 * (1 + 2 + 4 + 8 + 16) = 62? 
    # Actually Moore bound for cubic girth 2g: n >= 2 * (3^g - 3) / 2 ... 
    # Standard: n >= 2 * sum_{i=0}^{g-1} (k-1)^i for even girth 2g? 
    # For girth g (cycle length), radius r = g//2.
    # n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    # For k=3, g=16, r=8: n >= 2 * sum_{i=0}^{7} 2^i = 2*(2^8 - 1) = 510
    # But task says lower bound is 512. Let's start from 512.
    moore_bound = 512
    # We want to find the smallest n. Search upward.
    # Random 3-regular graphs: girth ~ log_2(n). For girth 16, need n ~ 2^16 = 65536?
    # That's too big. Actually girth of random d-regular is ~ log_{d-1}(n).
    # For d=3, girth ~ log_2(n). girth 16 means n ~ 2^16 = 65536.
    # Hmm, that's large. But the known best is 936. 
    # Wait, the task says smallest known is 936. So random search up to ~1000-2000.
    # Let's search n from moore_bound to some max, in steps.

    # Actually, let's be smarter. Try specific n values.
    # The known best is 936. Let's try to find something <= 936 or better.
    # Search n from 512 to 2000, step 2 (n must be even for 3-regular).
    
    found = False
    n = moore_bound
    while n <= 3000 and time.time() < deadline - 5:
        # Try a few random graphs for this n
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            edges = _random_3_reg_graph(n, seed=None)
            if edges is not None:
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    break
        if found:
            # Found a valid graph at this n, since we're going up, this is smallest.
            break
        n += 2

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: LUW construction (guaranteed valid, larger n)
    return _construct_luw(3, g)

def _random_3_reg_graph(n, seed=None):
    """Generate a random simple 3-regular graph of n vertices using configuration model.
    Returns None if impossible or if multi-edges/loops are generated."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if (u, v) in edges or (v, u) in edges:
            return None
        edges.add((u, v))
    return list(edges)

def _has_girth_at_least(n, edges, g):
    """Check if graph has no cycle shorter than g using non-backtracking BFS.
    Returns True if girth >= g, False otherwise."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each start vertex, do BFS but don't backtrack.
    # A cycle of length < g exists iff there's a non-backtracking walk of length < g that returns to start.
    # We check: for each start, BFS to depth g-1, tracking parent to avoid backtracking.
    # If we reach a vertex already visited at a shallower depth via a different path, 
    # and the combined path length < g, we have a short cycle.
    
    # More precisely: standard BFS from each start, but we must avoid the edge back to parent.
    # The girth is the shortest cycle. A cycle of length L can be found by BFS from any vertex on it.
    # Using BFS from start, if we find an edge (u,v) where dist[u] + dist[v] + 1 < g and 
    # v is not the parent of u (and u not parent of v), then we have a cycle of length dist[u]+dist[v]+1.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue  # Don't backtrack
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle: dist[u] + dist[v] + 1
                    # But we must ensure this isn't the tree edge (already handled by parent check)
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g):
    """Deterministic LUW construction fallback."""
    # For k=3, q=3. Girth ~ 2d+2. For g=16, need 2d+2 >= 16 => d >= 7.
    d = max(3, (g - 2) // 2)
    return _luw_internal(k, d)

def _luw_internal(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
