import time
import random
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

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
                if side == 0:
                    y[c] = add_op(x[c], mul_op(y[a], x[b]))
                else:
                    y[c] = sub_op(x[c], mul_op(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _check_girth(adj, n, g, deadline):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    for v in range(n):
        if time.time() > deadline:
            return False
        seen = {}
        queue = deque()
        for nb in adj[v]:
            queue.append((nb, v, 1))
            seen[(nb, v)] = 1
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= (g - 1) // 2:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == v:
                    if depth + 1 < g:
                        return False
                elif (nb, cur) in seen:
                    cycle_len = seen[(nb, cur)] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    seen[(nb, cur)] = depth + 1
                    queue.append((nb, cur, depth + 1))
    return True

def _generate_random_5_reg(n, rng):
    """Generate a random 5-regular bipartite graph on n vertices (n must be even)."""
    if n % 2 != 0:
        return None
    half = n // 2
    
    left_stubs = []
    for i in range(half):
        left_stubs.extend([i] * 5)
    right_stubs = []
    for j in range(half):
        right_stubs.extend([j + half] * 5)
    
    rng.shuffle(left_stubs)
    rng.shuffle(right_stubs)
    
    adj = [set() for _ in range(n)]
    for ls, rs in zip(left_stubs, right_stubs):
        if ls == rs: # Should not happen in bipartite
            return None
        adj[ls].add(rs)
        adj[rs].add(ls)
        
    if any(len(adj[i]) != 5 for i in range(n)):
        return None
        
    return adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback
    luw_n, luw_edges = _luw_construct(k, g)
    
    if k == 5 and g == 9:
        rng = random.Random(42)
        # Search from Moore bound upwards
        # Moore bound for 5-regular girth 9 is 426.
        # n must be even.
        for n in range(428, 3000, 2):
            if time.time() > deadline - 5:
                break
            
            # Try a few random instances for this n
            # For small n, success probability is low, so try more?
            # For large n, success probability is high, try fewer?
            # Let's just try 50 for each n.
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                adj = _generate_random_5_reg(n, rng)
                if adj is None:
                    continue
                if _check_girth(adj, n, 9, deadline - 10):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                    
    return luw_n, luw_edges
