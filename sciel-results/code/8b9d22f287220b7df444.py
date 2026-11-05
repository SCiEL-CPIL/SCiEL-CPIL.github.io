"""Cayley-graph construction for 4-regular girth-10 graph on abelian groups."""
import math
import numpy as np
from collections import deque

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    moore_bound = 243
    best_n = None
    best_edges = None
    
    # Phase 1: Try cyclic groups Z_m
    for m in range(moore_bound, 450):
        found = _search_cyclic(m, g)
        if found is not None:
            a, b = found
            best_n = m
            best_edges = _build_cyclic_edges(m, a, b)
            break
    
    # Phase 2: Try Z_a x Z_b if needed
    if best_n is None or best_n > 350:
        for a in range(2, 250):
            min_b = max(2, math.ceil(moore_bound / a))
            for b in range(min_b, 250):
                n = a * b
                if n > 450 or (best_n is not None and n >= best_n):
                    break
                found = _search_product_fast(a, b, g)
                if found is not None:
                    (u1, u2), (v1, v2) = found
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = _build_product_edges(a, b, u1, u2, v1, v2)
                    break
            if best_n is not None and a * min_b >= best_n:
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    return _construct_fallback(4, g)


def _check_girth_cyclic(m, a, b, g):
    """Check girth >= g for Z_m with S={a,-a,b,-b}."""
    S = [a, (-a) % m, b, (-b) % m]
    inv = [2, 3, 0, 1]
    for L in range(2, g):
        if _nb_walk_cyclic(m, S, inv, L):
            return False
    return True


def _nb_walk_cyclic(m, S, inv, L):
    """Check for non-backtracking closed walk of length L in Z_m."""
    def _dfs(depth, s, last, first):
        if depth == L:
            return s % m == 0 and inv[last] != first
        for ng in range(4):
            if ng != inv[last] and _dfs(depth + 1, s + S[ng], ng, first):
                return True
        return False
    for sg in range(4):
        if _dfs(1, S[sg], sg, sg):
            return True
    return False


def _search_cyclic(m, g):
    for a in range(1, m // 2 + 1):
        if 2 * a == m:
            continue
        for b in range(a + 1, m // 2 + 1):
            if 2 * b == m:
                continue
            if a + b == m or b - a == 0:
                continue
            if math.gcd(math.gcd(a, b), m) != 1:
                continue
            if _check_girth_cyclic(m, a, b, g):
                return (a, b)
    return None


def _search_product_fast(a, b, g):
    """Search Z_a x Z_b with limited generator search."""
    # Try generators with u1=1 or small values
    candidates_u = [(1, 0), (1, 1), (0, 1), (1, b-1), (a-1, 0), (a-1, 1)]
    candidates_v = [(1, 0), (1, 1), (0, 1), (1, b-1), (a-1, 0), (a-1, 1), (2, 0), (0, 2)]
    for u in candidates_u:
        if u[0] >= a or u[1] >= b:
            continue
        for v in candidates_v:
            if v[0] >= a or v[1] >= b:
                continue
            nu = ((a - u[0]) % a, (b - u[1]) % b)
            nv = ((a - v[0]) % a, (b - v[1]) % b)
            if u == v or u == nv or nu == v or nu == nv:
                continue
            if not _connected_product(a, b, u, v):
                continue
            S = [u, nu, v, nv]
            if _check_girth_product(a, b, S, g):
                return (u, v)
    return None


def _connected_product(a, b, u, v):
    visited = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    gens = [u, ((a-u[0])%a, (b-u[1])%b), v, ((a-v[0])%a, (b-v[1])%b)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in gens:
            nx, ny = (x + dx) % a, (y + dy) % b
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return len(visited) == a * b


def _check_girth_product(a, b, S, g):
    inv = [2, 3, 0, 1]
    for L in range(2, g):
        if _nb_walk_product(a, b, S, inv, L):
            return False
    return True


def _nb_walk_product(a, b, S, inv, L):
    def _dfs(depth, x, y, last, first):
        if depth == L:
            return x == 0 and y == 0 and inv[last] != first
        for ng in range(4):
            if ng != inv[last]:
                dx, dy = S[ng]
                if _dfs(depth + 1, (x + dx) % a, (y + dy) % b, ng, first):
                    return True
        return False
    for sg in range(4):
        if _dfs(1, S[sg][0], S[sg][1], sg, sg):
            return True
    return False


def _build_cyclic_edges(m, a, b):
    edge_set = set()
    for i in range(m):
        for s in [a, b]:
            j = (i + s) % m
            edge_set.add((min(i, j), max(i, j)))
    return sorted(edge_set)


def _build_product_edges(a, b, u1, u2, v1, v2):
    edge_set = set()
    for i in range(a):
        for j in range(b):
            idx = i * b + j
            for dx, dy in [(u1, u2), (v1, v2)]:
                ni, nj = (i + dx) % a, (j + dy) % b
                nidx = ni * b + nj
                edge_set.add((min(idx, nidx), max(idx, nidx)))
    return sorted(edge_set)


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
