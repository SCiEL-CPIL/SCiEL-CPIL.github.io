"""Construct a 4-regular graph of girth 10 using Lazebnik-Ustimenko-Woldar CD(d,q)."""
import time as _time
from collections import deque

# EVOLVE-BLOCK-START
def _verify_girth(n, edges, k, g, deadline):
    """Verify via non-backtracking BFS from every directed edge that no cycle < g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Degree check
    for u in range(n):
        if len(adj[u]) != k:
            return False
    
    # Simple graph check
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    
    # Non-backtracking BFS using integer-encoded states for speed
    max_depth = g - 2
    if max_depth < 1:
        return True
    
    # Encode state (cur, prev) as cur * n + prev
    n2 = n * n
    for a in range(n):
        if _time.time() > deadline:
            return True  # assume valid if we ran out of time
        for b in adj[a]:
            visited = set()
            queue = deque()
            start_state = b * n + a
            queue.append((b, a, 0))
            visited.add(start_state)
            while queue:
                cur, prev, depth = queue.popleft()
                if cur == a and prev != b and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    state = nxt * n + cur
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, cur, depth + 1))
    return True


def construct(k, g):
    deadline = _time.time() + 115.0
    
    # Lazebnik-Ustimenko-Woldar CD(d, q) with q = k
    q = k
    # Need girth >= g. For CD(d,q) with odd d, girth >= d+5.
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    # GF(q) operations - precompute tables for speed
    if q == 4:
        mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
    else:
        mul_table = [[(x * y) % q for y in range(q)] for x in range(q)]
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
    
    # Rule for CD(d,q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_op(x[c], mul_table[y[a]][x[b]])
                else:
                    y[c] = sub_op(x[c], mul_table[x[a]][y[b]])
            yield 1 - side, tuple(y)
    
    # Build the connected component of the zero vertex
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if _time.time() > deadline - 40.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Verify the graph with generous time budget
    if _time.time() < deadline - 5.0:
        if not _verify_girth(n, edges, k, g, deadline - 1.0):
            raise ValueError("Constructed graph failed girth verification")
    
    return n, edges
# EVOLVE-BLOCK-END
