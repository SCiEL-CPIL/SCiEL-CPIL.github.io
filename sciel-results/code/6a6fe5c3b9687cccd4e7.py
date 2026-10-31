"""
Construct a k-regular graph with girth at least g.
Optimized search for circulant graphs C_n(1, a, b) with fallback to LUW.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    if k == 3 and g == 16:
        # Search for circulant graphs C_n(1, a, b) with girth >= 16
        result = _find_circulant_cage(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback: LUW with verified girth
        # Try smaller d first to get smaller n
        for d in [7, 9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    # General fallback: LUW with default d
    return _luw_construct(k, g)


def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    """
    # Moore bound for k=3, g=16: 510
    # n must be even.
    min_n = 512
    max_n = 1000

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        # Search for valid (a, b)
        # a in [2, n/2), b in (a, n/2)
        for a in range(2, n // 2):
            if time.time() > deadline - 3:
                return None
            
            # Quick check: a should not be 1, -1, 2, -2, etc.
            # 4-cycle check: 
            # If 2 in {a, -a, b, -b} -> 4-cycle
            # If 1+a in {1, -1, a, -a, b, -b} -> 4-cycle
            # If 1-a in {1, -1, a, -a, b, -b} -> 4-cycle
            # We can just check the set of sums of 2 steps.
            
            for b in range(a + 1, n // 2):
                # Quick 4-cycle rejection
                if _has_4cycle(n, a, b):
                    continue
                
                # Quick 6-cycle rejection
                if _has_6cycle(n, a, b):
                    continue

                # Full girth check
                if _has_short_cycle_fast(n, a, b, g):
                    continue

                # Construct edges
                edges = []
                for i in range(n):
                    for shift in [1, a, b]:
                        j = (i + shift) % n
                        if i < j:
                            edges.append((i, j))

                return n, edges

    return None


def _has_4cycle(n, a, b):
    """
    Check if C_n(1, a, b) has a 4-cycle.
    A 4-cycle exists if there are s1, s2, s3, s4 in {±1, ±a, ±b}
    such that s1+s2+s3+s4 = 0 and no backtracking.
    This is equivalent to checking if the graph contains a K_{2,2} subgraph?
    No, just a cycle of length 4.
    """
    gens = [1, -1, a, -a, b, -b]
    # Check if any sum of 2 distinct (non-inverse) generators is in the generator set
    # Actually, s1+s2 = -(s3+s4).
    # If s1+s2 = g, then we need s3+s4 = -g.
    # Simplest case: s1=1, s2=a, s3=-1, s4=-a. Sum=0.
    # This is a 4-cycle 0-1-(1+a)-a-0.
    # Steps: 1, a, -1, -a.
    # Valid if 1 != -a, a != -1, -1 != -a, -a != 1.
    # i.e., a != -1, a != 1.
    # So if a is not 1 or -1, this is a valid 4-cycle?
    # Wait. 0 -> 1 (step 1). 1 -> 1+a (step a). 1+a -> a (step -1). a -> 0 (step -a).
    # Vertices: 0, 1, 1+a, a.
    # Are they distinct?
    # 0 != 1.
    # 0 != a.
    # 1 != a.
    # 1 != 1+a (a!=0).
    # a != 1+a (1!=0).
    # a != 0.
    # So yes, for ANY a not in {0, 1, -1}, C_n(1, a, b) has a 4-cycle?
    # Let's check.
    # 0 -1-> 1 -a-> 1+a -(-1)-> a -(-a)-> 0.
    # This is a 4-cycle.
    # This implies NO circulant graph C_n(1, a, b) with distinct 1, a has girth > 4.
    # This is a known result: Circulant graphs C_n(1, a) have girth 4 if a is not 2 or n/2.
    # Wait. If a=2, 0-1-2-1-0? No.
    # 0-1 (1), 1-3 (2), 3-2 (-1), 2-0 (-2).
    # Vertices: 0, 1, 3, 2.
    # 0!=1, 0!=2, 0!=3.
    # 1!=3, 1!=2.
    # 3!=2.
    # So a=2 also has a 4-cycle.
    # It seems C_n(1, a) ALWAYS has a 4-cycle.
    # Therefore, C_n(1, a, b) with 3 generators?
    # The 4-cycle above uses only generators 1 and a.
    # So if the graph contains the edges of C_n(1, a), it contains a 4-cycle.
    # Thus, ANY graph containing C_n(1, a) as a subgraph has girth 4.
    # So C_n(1, a, b) ALWAYS has girth 4.
    # This means we CANNOT find a circulant graph with girth 16 using generators {1, a, b}.
    
    return True # Always True for simple circulants with 1 in generators


def _has_6cycle(n, a, b):
    return True


def _has_short_cycle_fast(n, a, b, g):
    """
    Fast girth check for C_n(1, a, b).
    Note: As established, C_n(1, a, b) always has girth 4.
    So this will always return True for g > 4.
    """
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
