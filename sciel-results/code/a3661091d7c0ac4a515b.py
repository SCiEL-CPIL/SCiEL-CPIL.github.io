"""Program using algebraic voltage-graph lifts for high-girth regular graphs."""
# EVOLVE-BLOCK-START
from collections import deque
import math

def _check_girth_fast(n, edges, min_girth):
    """Check girth using non-backtracking walks; returns True if girth >= min_girth."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    half = min_girth // 2
    for start in range(n):
        # dist[node] = shortest non-backtracking distance from start
        dist = {start: 0}
        prev = {start: -1}
        queue = deque([(start, -1, 0)])
        while queue:
            node, par, d = queue.popleft()
            if d >= half:
                continue
            for nb in adj[node]:
                if nb == par:
                    continue
                if nb in dist:
                    cycle_len = d + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    dist[nb] = d + 1
                    prev[nb] = node
                    queue.append((nb, node, d + 1))
    return True

def _build_cayley_cyclic(n, generators):
    """Build a Cayley graph on Z_n with given generators (symmetric set)."""
    # generators should be symmetric: if g in gens, then -g in gens
    # and 0 not in gens
    seen = set()
    edges = []
    for u in range(n):
        for g in generators:
            v = (u + g) % n
            if v <= u:
                continue
            key = (u, v)
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return edges

def _check_cayley_girth(n, generators, min_girth):
    """Check if Cayley graph on Z_n has girth >= min_girth."""
    adj = [[] for _ in range(n)]
    for u in range(n):
        for g in generators:
            v = (u + g) % n
            adj[u].append(v)
    
    half = min_girth // 2
    for start in range(n):
        dist = {start: 0}
        prev = {start: -1}
        queue = deque([(start, -1, 0)])
        while queue:
            node, par, d = queue.popleft()
            if d >= half:
                continue
            for nb in adj[node]:
                if nb == par:
                    continue
                if nb in dist:
                    cycle_len = d + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    dist[nb] = d + 1
                    prev[nb] = node
                    queue.append((nb, node, d + 1))
    return True

def _find_cayley_4reg_girth10():
    """Search for a 4-regular Cayley graph on Z_n with girth >= 10."""
    # For 4-regular, we need 2 generators (and their negatives)
    # Try various n and generator pairs
    # Moore bound ~243, known best 320
    # Try n values that are multiples of small numbers for good structure
    
    # Strategy: try n from 244 to 320, with generators (a, b) where gcd(a,b,n)=1
    # and the graph is connected and has no short cycles
    
    candidates_n = list(range(244, 321))
    # Also try some structured n values
    for n in candidates_n:
        # Try generator pairs (a, b) with 1 <= a < b < n//2
        # To keep it efficient, limit search
        for a in range(1, min(n // 2, 50)):
            for b in range(a + 1, min(n // 2, 50)):
                if math.gcd(a, b, n) != 1:
                    continue
                generators = [a, -a % n, b, -b % n]
                # Check it's 4-regular (should be by construction if a != b and a != -b mod n)
                if a == b or a == (-b) % n:
                    continue
                if _check_cayley_girth(n, generators, 10):
                    edges = _build_cayley_cyclic(n, generators)
                    return n, edges
    return None

def construct(k, g):
    if k == 4 and g == 10:
        result = _find_cayley_4reg_girth10()
        if result is not None:
            return result
        # Fallback: use the L-U-W style algebraic construction
        return _luw_construct(4, 10)
    else:
        return _luw_construct(k, g)

def _luw_construct(k, g):
    """L-U-W style algebraic construction for general k, g."""
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
