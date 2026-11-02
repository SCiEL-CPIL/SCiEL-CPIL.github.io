import time
import math
import random

# EVOLVE-BLOCK-START

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    d = max(3, d)

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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _check_girth_bfs(n, adj, g):
    """Check if graph has girth >= g using BFS from each vertex (non-backtracking)."""
    for s in range(n):
        dist = [(-1, -1)] * n
        dist[s] = (0, -1)
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            du, pu = dist[u]
            if du + 1 >= g:
                continue
            for v in adj[u]:
                if v == pu:
                    continue
                if dist[v][0] == -1:
                    dist[v] = (du + 1, u)
                    queue.append(v)
                elif dist[v][0] > du + 1:
                    cycle_len = dist[v][0] + du + 1
                    if cycle_len < g:
                        return False
                    dist[v] = (du + 1, u)
                    queue.append(v)
                else:
                    if dist[v][1] != u:
                        cycle_len = dist[v][0] + du + 1
                        if cycle_len < g:
                            return False
    return True

def _check_girth_fast(n, adj, g):
    """
    Optimized girth check.
    Since the graph is vertex transitive (for circulants/dihedral), we only need to check from 0.
    However, for safety, we check from 0. If it fails, return False.
    If it passes, is it guaranteed?
    For vertex transitive graphs, yes.
    """
    s = 0
    # state: (node, prev_node)
    # dist: dictionary or array?
    # n is small (~1000). Array of size n * 3?
    # prev_node can be -1, 0..n-1.
    # We can use a dict for visited states to be safe, or a 2D array if we map prev_node.
    # Actually, just a set of tuples is fine for n=1000, depth=15.
    
    visited = set()
    # (node, prev)
    queue = [(0, -1, 0)]
    visited.add((0, -1))
    
    qi = 0
    while qi < len(queue):
        u, p, d = queue[qi]
        qi += 1
        
        if u == 0 and d > 0 and d < g:
            return False
        
        if d + 1 >= g:
            continue
            
        for v in adj[u]:
            if v == p:
                continue
            state = (v, u)
            if state not in visited:
                visited.add(state)
                queue.append((v, u, d + 1))
                
    return True


def _search_circulant(n, a, g):
    """
    Check if C(n; {a, n-a, n/2}) has girth >= g.
    Returns True if valid.
    """
    if n % 2 != 0:
        return False
    h = n // 2
    if math.gcd(n, a, h) != 1:
        return False
    if a == 0 or a == h:
        return False
    if 2 * a % n == 0:
        return False
        
    # Build adjacency list for vertex 0 to check girth?
    # No, build full adj list for the check function.
    # Or implement specific check.
    # Let's build adj list.
    adj = [[] for _ in range(n)]
    for u in range(n):
        adj[u].append((u + a) % n)
        adj[u].append((u - a) % n)
        adj[u].append((u + h) % n)
        
    return _check_girth_fast(n, adj, g)


def _search_dihedral(n, a, g):
    """
    Check if Cay(D_n, {s, r^a, r^-a}) has girth >= g.
    n is the order of rotation. Total vertices = 2n.
    """
    if math.gcd(n, a) != 1:
        return False
    if 2 * a % n == 0:
        return False
        
    total_n = 2 * n
    adj = [[] for _ in range(total_n)]
    
    for i in range(n):
        # Node i is r^i
        adj[i].append(n + i) # s
        adj[i].append((i + a) % n) # r^a
        adj[i].append((i - a) % n) # r^-a
        
    for i in range(n):
        # Node n+i is s r^i
        adj[n + i].append((-i) % n) # s * s = r^-i
        adj[n + i].append(n + (i + a) % n) # s r^i r^a = s r^(i+a)
        adj[n + i].append(n + (i - a) % n) # s r^i r^-a = s r^(i-a)
        
    return _check_girth_fast(total_n, adj, g)


def construct(k, g):
    deadline = time.time() + 110
    
    if k == 3:
        # Moore bound for k=3, even g
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
        if moore_bound % 2 != 0:
            moore_bound += 1
            
        # Search for smallest n
        # We search total vertices N
        # N must be even for cyclic (n/2)
        # N must be even for dihedral (2n)
        
        # Limit search to avoid timeout
        # Moore bound is 512 for g=16.
        # Let's search up to 2000.
        max_n_search = 2000
        
        for N in range(moore_bound, max_n_search + 1, 2):
            if time.time() > deadline - 5:
                break
                
            # Try Cyclic C(N; {a, N-a, N/2})
            h = N // 2
            # a must be coprime with gcd(N, h) = h?
            # gcd(N, a, h) = 1. Since h = N/2, gcd(N, h) = h.
            # So gcd(h, a) must be 1.
            for a in range(1, h):
                if math.gcd(h, a) != 1:
                    continue
                if _search_circulant(N, a, g):
                    edges = []
                    for u in range(N):
                        for v in ((u + a) % N, (u - a) % N, (u + h) % N):
                            if u < v:
                                edges.append((u, v))
                    return N, edges
                    
            # Try Dihedral Cay(D_{N/2}, {s, r^a, r^-a})
            # Total vertices = 2 * (N/2) = N.
            # So rotation order n_rot = N // 2.
            n_rot = N // 2
            for a in range(1, n_rot):
                if _search_dihedral(n_rot, a, g):
                    edges = []
                    # Rebuild edges for dihedral
                    for i in range(n_rot):
                        u1 = i
                        v1 = n_rot + i
                        if u1 < v1: edges.append((u1, v1))
                        
                        v2 = (i + a) % n_rot
                        if u1 < v2: edges.append((u1, v2))
                        
                        v3 = (i - a) % n_rot
                        if u1 < v3: edges.append((u1, v3))
                        
                        u2 = n_rot + i
                        v4 = (-i) % n_rot
                        if u2 < v4: edges.append((u2, v4))
                        
                        v5 = n_rot + (i + a) % n_rot
                        if u2 < v5: edges.append((u2, v5))
                        
                        v6 = n_rot + (i - a) % n_rot
                        if u2 < v6: edges.append((u2, v6))
                    return N, edges

    # Fallback to LUW
    try:
        return _luw_construct(k, g, deadline)
    except:
        return 0, []

# EVOLVE-BLOCK-END
