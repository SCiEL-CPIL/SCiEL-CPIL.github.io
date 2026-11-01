"""Rewritten design: random 3-regular graphs + local edge-swap repair to push girth up."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _moore_lower_bound(k, g):
    # Moore bound for girth g (even) with degree k
    # n >= k * (1 + (k-1) + ... + (k-1)^(g/2 - 1))
    if g % 2 == 0:
        h = g // 2
        total = 0
        term = 1
        for i in range(h):
            total += term
            term *= (k - 1)
        return k * total
    else:
        h = (g - 1) // 2
        total = 1
        term = 1
        for i in range(h):
            term *= (k - 1)
            total += term
        return k * total + (k - 1) * term

def _compute_girth(adj):
    """Compute girth using BFS from each vertex (non-backtracking)."""
    n = len(adj)
    girth = float('inf')
    for start in range(n):
        # BFS tracking (vertex, parent)
        dist = {}
        queue = [(start, -1)]
        dist[(start, -1)] = 0
        while queue:
            new_queue = []
            for v, parent in queue:
                d = dist[(v, parent)]
                for u in adj[v]:
                    if u == parent:
                        continue
                    if (u, v) in dist:
                        # cycle found of length d + 1
                        girth = min(girth, d + 1)
                    elif (u, v) not in dist:
                        dist[(u, v)] = d + 1
                        new_queue.append((u, v))
            queue = new_queue
            if girth <= 2 * d:  # can't find shorter cycle from this start
                break
    return girth

def _generate_random_3_regular(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    # 3n stubs
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    # Pair up stubs
    adj = [[] for _ in range(n)]
    valid = True
    for i in range(0, 3 * n, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].append(v)
        adj[v].append(u)
    if not valid:
        return None
    return adj

def _edge_swap(adj, n):
    """Perform a random edge swap: remove (u,v) and (w,x), add (u,w) and (v,x)."""
    # Collect all edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    if len(edges) < 2:
        return False
    # Pick two random edges
    i, j = random.sample(range(len(edges)), 2)
    u, v = edges[i]
    w, x = edges[j]
    # Check no conflicts
    if len({u, v, w, x}) < 4:
        return False
    # Check new edges don't already exist
    if w in adj[u] or x in adj[v]:
        return False
    # Perform swap
    adj[u].remove(v)
    adj[v].remove(u)
    adj[w].remove(x)
    adj[x].remove(w)
    adj[u].append(w)
    adj[w].append(u)
    adj[v].append(x)
    adj[x].append(v)
    return True

def _find_short_cycles(adj, g_target):
    """Find cycles shorter than g_target. Returns list of cycles (list of vertices)."""
    n = len(adj)
    cycles = []
    for start in range(n):
        # BFS with parent tracking
        dist = {}
        parent = {}
        queue = [(start, -1)]
        dist[(start, -1)] = 0
        while queue:
            new_queue = []
            for v, par in queue:
                d = dist[(v, par)]
                if d >= g_target - 1:
                    continue
                for u in adj[v]:
                    if u == par:
                        continue
                    if (u, v) in dist:
                        # Found a cycle
                        cycle_len = d + 1
                        if cycle_len < g_target:
                            # Reconstruct cycle
                            cycle = [v]
                            cur = v
                            while cur != start:
                                cur = parent.get((cur, par), None)
                                if cur is None:
                                    break
                                cycle.append(cur)
                                par = par  # simplified
                            # Just record the length
                            cycles.append(cycle_len)
                    elif (u, v) not in dist:
                        dist[(u, v)] = d + 1
                        parent[(u, v)] = (v, par)
                        new_queue.append((u, v))
            queue = new_queue
    return cycles

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    deadline = time.time() + 110  # safety margin
    
    # Moore bound
    moore_lb = _moore_lower_bound(k, g)
    
    # Start with a reasonable size
    # For k=3, g=16: Moore bound is 512. Known best is 936.
    # Try sizes from near Moore bound upward.
    
    if k != 3:
        # Fallback to LUW-style for other k
        return _luw_construct(k, g)
    
    # For k=3, use random + local search
    best_adj = None
    best_n = None
    best_girth = 0
    
    # Try several sizes
    for n in range(moore_lb, moore_lb + 400, 8):
        if time.time() > deadline - 5:
            break
        if n % 2 != 0:
            n += 1
        
        # Try to generate and repair
        success = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            adj = _generate_random_3_regular(n)
            if adj is None:
                continue
            
            # Check girth
            girth = _compute_girth(adj)
            if girth >= g:
                best_adj = adj
                best_n = n
                success = True
                break
            
            # If girth is close, try to improve with edge swaps
            if girth >= g - 4:
                # Local search: try edge swaps to increase girth
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    if _edge_swap(adj, n):
                        new_girth = _compute_girth(adj)
                        if new_girth > girth:
                            girth = new_girth
                            if girth >= g:
                                best_adj = adj
                                best_n = n
                                success = True
                                break
                if success:
                    break
        
        if best_adj is not None and best_n <= n:
            # Found a valid graph, try smaller n
            if best_n == n:
                break
            # Continue trying smaller n
            if n <= moore_lb + 20:
                break
    
    if best_adj is not None:
        edges = []
        for u in range(best_n):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return best_n, edges
    
    # Fallback: return LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar style construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
# EVOLVE-BLOCK-END
