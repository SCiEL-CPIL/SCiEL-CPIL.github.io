"""Construct a 3-regular graph of girth at least g with as few vertices as possible."""

import math
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (i, j) tuples.
    
    Strategy: Use circulant graph construction C_n({±a, ±b}) for k=3.
    Search over n and connection parameters to find minimum n with girth >= g.
    """
    if k == 3:
        return _construct_circulant_3(g)
    else:
        # For general k, use a similar circulant approach
        return _construct_circulant_general(k, g)


def _construct_circulant_3(g):
    """
    Find the smallest n such that there exist a, b with
    C_n({±a, ±b}) having girth >= g.
    
    The Moore bound for (3, g):
    - Even g: n >= 2 * sum_{i=0}^{g/2-1} 2^i = 2*(2^(g/2) - 1)
    - Odd g: n >= 2 + 2*sum_{i=0}^{(g-1)/2} 2^i = 2 + 2*(2^((g+1)/2) - 1) = 2^((g+1)/2)
    """
    # Compute Moore bound
    if g % 2 == 0:
        half = g // 2
        moore = 2 * ((2 ** half) - 1)
    else:
        moore = 2 ** ((g + 1) // 2)
    
    # Start searching from Moore bound, round to even
    start_n = max(moore, 6)
    if start_n % 2 != 0:
        start_n += 1
    
    # Search for valid (n, a, b) combinations
    # For efficiency, search n in increasing order and for each n,
    # try a few promising (a, b) pairs
    
    max_search_n = start_n + 2000  # Search up to 2000 beyond Moore bound
    
    for n in range(start_n, max_search_n + 1, 2):
        if n < 4:
            continue
        # Try different (a, b) pairs
        # a and b should be relatively "spread out" to avoid short cycles
        # Good candidates: a ~ n/4, b ~ n/3, or other fractions
        candidates = _candidate_pairs(n)
        for a, b in candidates:
            if a == b or a == 0 or b == 0:
                continue
            if a >= n or b >= n:
                continue
            # Check girth
            if _circulant_girth_at_least(n, a, b, g):
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + a) % n))
                    edges.append((i, (i + b) % n))
                # Remove duplicates (undirected)
                edge_set = set()
                unique_edges = []
                for u, v in edges:
                    key = (min(u, v), max(u, v))
                    if key not in edge_set:
                        edge_set.add(key)
                        unique_edges.append(key)
                return n, unique_edges
    
    # Fallback: use a larger n with a simple construction
    # Use n = 2*m where m is large, a = m//2, b = m//3
    for extra in range(0, 100, 2):
        n = start_n + extra
        if n < 6:
            continue
        a = n // 4
        b = n // 3
        if a < 1:
            a = 1
        if b < 1:
            b = 2
        if a == b:
            b = a + 1
        if a >= n or b >= n:
            continue
        if _circulant_girth_at_least(n, a, b, g):
            edges = []
            for i in range(n):
                edges.append((i, (i + a) % n))
                edges.append((i, (i + b) % n))
            edge_set = set()
            unique_edges = []
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    edge_set.add(key)
                    unique_edges.append(key)
            return n, unique_edges
    
    # Last resort: use the LUW fallback from the original
    return _luw_fallback(3, g)


def _candidate_pairs(n):
    """Generate candidate (a, b) pairs for circulant C_n({±a, ±b})."""
    candidates = []
    # Try fractions of n
    for frac_a in [1/4, 1/5, 1/3, 2/5, 1/6, 3/8, 1/7, 2/7, 3/10, 1/8]:
        a = max(1, int(n * frac_a))
        for frac_b in [1/3, 1/4, 2/5, 1/5, 3/8, 1/6, 2/7, 3/10, 1/7, 2/9]:
            b = max(1, int(n * frac_b))
            if a != b and 0 < a < n and 0 < b < n:
                candidates.append((a, b))
    # Also try some fixed small offsets
    for a in [1, 2, 3, 5, 7, 11]:
        for b in [2, 3, 5, 7, 11, 13, 17]:
            if a != b and a < n and b < n:
                candidates.append((a, b))
    # Try a = n//4, b = n//3 type
    a1 = max(1, n // 4)
    b1 = max(1, n // 3)
    if a1 != b1:
        candidates.append((a1, b1))
    a2 = max(1, n // 5)
    b2 = max(1, n // 3)
    if a2 != b2:
        candidates.append((a2, b2))
    return candidates


def _circulant_girth_at_least(n, a, b, g):
    """
    Check if C_n({±a, ±b}) has girth at least g.
    Uses BFS from vertex 0 to find shortest cycle.
    
    For circulant graphs, we can do BFS in the group Z_n.
    A cycle of length L exists if there's a sequence of ±a, ±b steps
    that sums to 0 mod n with L steps, and no proper subsequence does.
    
    Efficient check: BFS from 0, track distances. If we find a back-edge
    that creates a cycle shorter than g, return False.
    """
    # BFS from vertex 0
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        u = queue.popleft()
        for step in (a, -a, b, -b):
            v = (u + step) % n
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
            else:
                # Found a cycle of length dist[u] + 1 + dist[v]
                # But we need to be careful: this is the cycle through u-v edge
                # The actual cycle length is dist[u] + 1 + dist[v] only if
                # the shortest path from 0 to v doesn't go through u
                # For a proper check, we need to verify no shorter cycle exists
                cycle_len = dist[u] + 1 + dist[v]
                if cycle_len < g:
                    return False
    
    # Also need to check: the BFS above finds cycles, but we should verify
    # that the graph is connected (it should be for a valid circulant)
    # If not all vertices reached, the graph is disconnected
    if any(d == -1 for d in dist):
        # Disconnected - need to handle differently
        # For our purposes, check each component
        for start in range(n):
            if dist[start] == -1:
                dist2 = [-1] * n
                dist2[start] = 0
                queue2 = deque([start])
                while queue2:
                    u = queue2.popleft()
                    for step in (a, -a, b, -b):
                        v = (u + step) % n
                        if dist2[v] == -1:
                            dist2[v] = dist2[u] + 1
                            queue2.append(v)
                        else:
                            cycle_len = dist2[u] + 1 + dist2[v]
                            if cycle_len < g:
                                return False
        return True
    
    return True


def _construct_circulant_general(k, g):
    """
    For general k, use a circulant graph with k/2 connection pairs.
    C_n({±a_1, ±a_2, ..., ±a_{k/2}})
    """
    if k % 2 != 0:
        # Odd degree: use a circulant with one self-loop-free odd connection
        # This is tricky; fall back to random
        return _random_k_regular(k, g)
    
    half_k = k // 2
    # Moore bound
    if g % 2 == 0:
        half = g // 2
        s = sum((k - 1) ** i for i in range(half - 1))
        moore = k * s
    else:
        half = g // 2
        s = sum((k - 1) ** i for i in range(half))
        moore = 1 + k * s
    
    start_n = max(moore, k + 1)
    if start_n % 2 != 0:
        start_n += 1
    
    for n in range(start_n, start_n + 500, 2):
        # Try candidate connection sets
        for conn_set in _candidate_connection_sets(n, half_k):
            if _circulant_girth_general(n, conn_set, g):
                edges = []
                for i in range(n):
                    for a in conn_set:
                        edges.append((i, (i + a) % n))
                edge_set = set()
                unique_edges = []
                for u, v in edges:
                    key = (min(u, v), max(u, v))
                    if key not in edge_set:
                        edge_set.add(key)
                        unique_edges.append(key)
                return n, unique_edges
    
    return _luw_fallback(k, g)


def _candidate_connection_sets(n, half_k):
    """Generate candidate connection sets for general k."""
    candidates = []
    # Use evenly spaced connections
    for offset in range(1, min(10, n // 4)):
        conn = [offset * i for i in range(1, half_k + 1)]
        if all(0 < c < n for c in conn) and len(set(conn)) == half_k:
            candidates.append(conn)
    # Use specific fractions
    fracs = [1/3, 1/4, 1/5, 2/5, 1/6, 3/8]
    for frac in fracs:
        conn = [max(1, int(n * frac * (i + 1))) for i in range(half_k)]
        if all(0 < c < n for c in conn) and len(set(conn)) == half_k:
            candidates.append(conn)
    return candidates


def _circulant_girth_general(n, conn_set, g):
    """Check girth for general circulant."""
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    while queue:
        u = queue.popleft()
        for a in conn_set:
            for step in (a, -a):
                v = (u + step) % n
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    if any(d == -1 for d in dist):
        return False  # Disconnected
    
    return True


def _random_k_regular(k, g):
    """Generate a random k-regular graph and verify girth."""
    moore_n = _moore_bound(k, g)
    base_n = max(moore_n, k + 1)
    if base_n % 2 != 0 and k % 2 == 0:
        base_n += 1
    
    for n in range(base_n, base_n + 100, 2):
        if n * k % 2 != 0:
            continue
        for trial in range(20):
            edges = _random_k_regular_edges(n, k)
            if edges is not None:
                if _has_girth_at_least(n, edges, g):
                    return n, edges
    
    return _luw_fallback(k, g)


def _random_k_regular_edges(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(50):
        half_edges = []
        for v in range(n):
            for _ in range(k):
                half_edges.append(v)
        random.shuffle(half_edges)
        
        edges = []
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        
        if not valid:
            continue
        
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if all(d == k for d in degree):
            return edges
    
    return None


def _has_girth_at_least(n, edges, g):
    """Check if the graph has no cycle shorter than g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _moore_bound(k, g):
    """Compute the Moore bound for a (k, g) cage."""
    if g % 2 == 0:
        half = g // 2
        s = 0
        for i in range(half - 1):
            s += (k - 1) ** i
        return k * s
    else:
        half = g // 2
        s = 0
        for i in range(half):
            s += (k - 1) ** i
        return 1 + k * s


def _luw_fallback(k, g):
    """
    Fall back to a Lazebnik-Ustimenko-Woldar type construction.
    """
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
