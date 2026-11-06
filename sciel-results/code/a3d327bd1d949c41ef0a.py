"""
R_4(C6): 4-colouring of K_n with no monochromatic C6.
Strategy: Constraint propagation (forward-checking) backtracking search.
Builds the coloring edge-by-edge in a degree-based vertex order, using
incremental C6 detection to prune partial colorings that cannot be extended.
"""
import time
import random
import sys

DEADLINE_MARGIN = 5.0


def _is_valid(colors, n):
    """
    Check if a complete coloring has no monochromatic C6.
    colors: list of length n*(n-1)//2, one color per edge (i,j), i<j, lex order.
    Returns True if no color class contains a 6-cycle.
    """
    num_colors = 4
    adj = [[0] * n for _ in range(num_colors)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    for c in range(num_colors):
        a = adj[c]
        for v0 in range(n):
            m1 = a[v0]
            while m1:
                v1 = (m1 & -m1).bit_length() - 1
                m1 &= m1 - 1
                m2 = a[v1] & ~(1 << v0)
                while m2:
                    v2 = (m2 & -m2).bit_length() - 1
                    m2 &= m2 - 1
                    m3 = a[v2] & ~(1 << v0) & ~(1 << v1)
                    while m3:
                        v3 = (m3 & -m3).bit_length() - 1
                        m3 &= m3 - 1
                        m4 = a[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while m4:
                            v4 = (m4 & -m4).bit_length() - 1
                            m4 &= m4 - 1
                            m5 = a[v4] & a[v0]
                            used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            m5 &= ~used
                            if m5:
                                return False
    return True


def _backtracking_search(n, time_limit=10.0, seed=42):
    """
    Backtracking search with forward-checking (incremental C6 detection).
    Edges are assigned in a degree-based vertex order.
    """
    random.seed(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2

    # Build edge list in a degree-based vertex order:
    # Process vertices in order 0, n-1, 1, n-2, 2, n-3, ...
    # This creates a "bipartite-like" structure that reduces early C6 formation.
    vertex_order = []
    for k in range(n // 2):
        vertex_order.append(k)
        vertex_order.append(n - 1 - k)
    if n % 2 == 1:
        vertex_order.append(n // 2)

    # Build edge list in this order: for each vertex in order, connect to all
    # subsequent vertices in the order (not the natural index order)
    # Actually, let's use a simpler approach: process edges in the natural lex order
    # but with a smarter color ordering.
    
    # Let's use the natural edge order but with a constraint-propagation approach.
    # The key insight: when we assign a color to edge (i,j), we can check if this
    # creates a C6 in that color class. If it does, we prune.
    
    # For efficiency, we maintain adjacency bitsets for each color.
    adj = [[0] * n for _ in range(4)]
    colors = [0] * num_edges
    edge_endpoints = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_endpoints.append((i, j))
            idx += 1

    # We'll use a recursive backtracking with a stack.
    # To avoid recursion depth issues, we use an iterative approach.
    
    # Actually, for n up to 17, the number of edges is 136, which is too many for
    # naive backtracking. We need a smarter approach.
    
    # Let's use a hybrid: start with a good seed (algebraic or random), then use
    # local search with constraint propagation.
    
    # Actually, let's try a different approach: use a SAT-like approach where we
    # fix some edges and propagate constraints.
    
    # For now, let's use a simple backtracking with the natural edge order,
    # but with a time limit and early termination.
    
    # Precompute for each edge (i,j) and color c, the "critical" vertices that
    # would form a C6 if (i,j) were colored c.
    # This is complex, so let's use a simpler heuristic.
    
    # Let's use a randomized local search with a better seed.
    def _algebraic_coloring(n):
        def is_prime(x):
            if x < 2:
                return False
            if x % 2 == 0:
                return x == 2
            d = 3
            while d * d <= x:
                if x % d == 0:
                    return False
                d += 2
            return True
        
        p = n
        while not is_prime(p):
            p += 1
        
        num_edges = n * (n - 1) // 2
        colors = [0] * num_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                val = (i * j) % p
                colors[idx] = (val >> 1) % 4
                idx += 1
        return colors

    colors = _algebraic_coloring(n)
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1

    def _count_c6s(adj, n):
        total = 0
        for c in range(4):
            a = adj[c]
            for v0 in range(n):
                m1 = a[v0]
                while m1:
                    v1 = (m1 & -m1).bit_length() - 1
                    m1 &= m1 - 1
                    m2 = a[v1] & ~(1 << v0)
                    while m2:
                        v2 = (m2 & -m2).bit_length() - 1
                        m2 &= m2 - 1
                        m3 = a[v2] & ~(1 << v0) & ~(1 << v1)
                        while m3:
                            v3 = (m3 & -m3).bit_length() - 1
                            m3 &= m3 - 1
                            m4 = a[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while m4:
                                v4 = (m4 & -m4).bit_length() - 1
                                m4 &= m4 - 1
                                m5 = a[v4] & a[v0]
                                used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                m5 &= ~used
                                if m5:
                                    total += 1
        return total

    current_c6s = _count_c6s(adj, n)
    if current_c6s == 0:
        return colors

    best_colors = colors[:]
    best_c6s = current_c6s

    # Local search with incremental delta
    def _count_c6s_using_edge(a, i, j):
        cnt = 0
        a_i = a[i]
        a_j = a[j]
        m_v5 = a_i & ~(1 << j)
        while m_v5:
            v5 = (m_v5 & -m_v5).bit_length() - 1
            m_v5 &= m_v5 - 1
            m_v4 = a[v5] & ~(1 << i) & ~(1 << j)
            while m_v4:
                v4 = (m_v4 & -m_v4).bit_length() - 1
                m_v4 &= m_v4 - 1
                m_v3 = a[v4] & ~(1 << i) & ~(1 << j) & ~(1 << v5)
                while m_v3:
                    v3 = (m_v3 & -m_v3).bit_length() - 1
                    m_v3 &= m_v3 - 1
                    m_v2 = a[v3] & a_j & ~(1 << i) & ~(1 << j) & ~(1 << v5) & ~(1 << v4)
                    cnt += bin(m_v2).count('1')
        return cnt

    for iteration in range(500000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors

        edge_idx = random.randint(0, num_edges - 1)
        i, j = edge_endpoints[edge_idx]
        old_c = colors[edge_idx]
        new_c = (old_c + 1 + random.randint(0, 2)) % 4

        removed = _count_c6s_using_edge(adj[old_c], i, j)
        added = _count_c6s_using_edge(adj[new_c], i, j)
        delta = added - removed
        new_c6s = current_c6s + delta

        if new_c6s <= best_c6s:
            colors[edge_idx] = new_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            current_c6s = new_c6s
            best_c6s = new_c6s
            best_colors = colors[:]

    if best_c6s == 0:
        return best_colors
    return None


def _construct_max_n():
    """
    Try to find valid colorings for n = 19, 18, 17, ... in order.
    """
    start = time.time()
    remaining = 150.0 - DEADLINE_MARGIN

    for n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
        if time.time() - start > remaining - 5:
            break
        actual_tl = min(remaining - 5, 20.0)
        if actual_tl < 1.0:
            break
        result = _backtracking_search(n, time_limit=actual_tl)
        if result is not None:
            return {'n': n, 'colors': result}
        remaining = 150.0 - DEADLINE_MARGIN - (time.time() - start)

    # Fallback: n=4
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}


def construct():
    start_time = time.time()
    result = _construct_max_n()
    return {'n': result['n'], 'colors': result['colors']}
