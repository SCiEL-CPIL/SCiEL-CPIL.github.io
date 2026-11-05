"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np
import math

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
    
    # Target: n=320
    # Try group Z_80 x Z_2 x Z_2
    n_target = 320
    found = _search_cayley_z80x2x2(n_target, g)
    if found:
        return n_target, found
    
    # Fallback: Search other abelian groups of order 320 or nearby
    # Factors of 320: 2^6 * 5
    # Groups: Z_320, Z_160xZ_2, Z_80xZ_4, Z_80xZ_2xZ_2, Z_40xZ_8, etc.
    
    # If not found, use fallback
    return _construct_fallback(4, g)

def _search_cayley_z80x2x2(n, g):
    """
    Search for a 4-regular girth-g Cayley graph on Z_80 x Z_2 x Z_2.
    """
    m1, m2, m3 = 80, 2, 2
    
    # Generate all elements
    # Element is (a, b, c) with a in 0..79, b in 0..1, c in 0..1
    # Inverse of (a, b, c) is (-a mod 80, b, c) since b,c are involutions
    # Wait, (a,b,c) + (a',b',c') = (a+a', b+b', c+c')
    # Inverse of (a,b,c) is (80-a, b, c) if we view Z2 as {0,1} with +.
    # Actually, in Z2, x = -x. So inverse is (80-a, b, c).
    
    # We need a set S of 4 elements such that:
    # 1. S is symmetric: s in S => -s in S.
    # 2. 0 not in S.
    # 3. s != -s for all s in S (no involutions in S? No, if s=-s, it's an involution.
    #    If s is an involution, s=-s. Then s is its own inverse.
    #    The set S must have 4 distinct elements.
    #    If s is an involution, s and -s are the same element.
    #    So we can have at most 2 involutions in S?
    #    If S = {u, v, w, x} and u=-u, then u is in S.
    #    If we have 2 involutions u, v, and a pair {w, -w}, then S={u, v, w, -w}.
    #    If we have 4 involutions, S={u, v, w, x} all self-inverse.
    
    # Strategy:
    # Case 1: 4 involutions.
    # Case 2: 2 involutions, 1 pair.
    # Case 3: 0 involutions, 2 pairs.
    
    # We will sample random sets.
    
    import random
    random.seed(42)
    
    all_elements = [(a, b, c) for a in range(80) for b in range(2) for c in range(2)]
    # Exclude identity
    non_identity = [e for e in all_elements if e != (0, 0, 0)]
    
    def get_inverse(e):
        a, b, c = e
        return ((-a) % 80, b, c)
    
    def is_involution(e):
        return e == get_inverse(e)
    
    # Pre-calculate inverses
    inv_map = {e: get_inverse(e) for e in all_elements}
    
    # We need to pick 4 elements.
    # To ensure symmetry, we pick pairs.
    
    # Case 2: 2 involutions + 1 pair
    involutions = [e for e in non_identity if is_involution(e)]
    # Involutions in Z_80 x Z_2 x Z_2:
    # 2a = 0 mod 80 => a in {0, 40}.
    # b, c can be anything.
    # So involutions are (0, b, c) and (40, b, c).
    # Total 8 involutions.
    
    pairs = []
    seen = set()
    for e in non_identity:
        if is_involution(e):
            continue
        inv_e = inv_map[e]
        if e < inv_e: # Canonical order
            pairs.append((e, inv_e))
    
    # Try random combinations
    for _ in range(200):
        # Pick 2 involutions
        if len(involutions) < 2:
            continue
        u, v = random.sample(involutions, 2)
        # Pick 1 pair
        if not pairs:
            continue
        w, w_inv = random.choice(pairs)
        
        S = [u, v, w, w_inv]
        
        # Check distinctness
        if len(set(S)) != 4:
            continue
        
        # Build graph and check girth
        edges = _build_edges_from_generators(S, m1, m2, m3)
        if _check_girth_numpy(edges, n, g):
            return edges
    
    # Case 3: 2 pairs
    for _ in range(200):
        if len(pairs) < 2:
            continue
        p1, p2 = random.sample(pairs, 2)
        S = [p1[0], p1[1], p2[0], p2[1]]
        if len(set(S)) != 4:
            continue
        
        edges = _build_edges_from_generators(S, m1, m2, m3)
        if _check_girth_numpy(edges, n, g):
            return edges
            
    return None

def _build_edges_from_generators(S, m1, m2, m3):
    n = m1 * m2 * m3
    edges = set()
    # Map (a,b,c) to index
    # index = a * m2 * m3 + b * m3 + c
    def idx(a, b, c):
        return a * m2 * m3 + b * m3 + c
    
    for a in range(m1):
        for b in range(m2):
            for c in range(m3):
                u = idx(a, b, c)
                for s in S:
                    sa, sb, sc = s
                    va = (a + sa) % m1
                    vb = (b + sb) % m2
                    vc = (c + sc) % m3
                    v = idx(va, vb, vc)
                    if u < v:
                        edges.add((u, v))
                    else:
                        edges.add((v, u))
    return list(edges)

def _check_girth_numpy(edges, n, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for u in range(n):
        if len(adj[u]) != 4:
            return False
            
    state_to_idx = {}
    idx = 0
    for u in range(n):
        for v in adj[u]:
            state_to_idx[(u, v)] = idx
            idx += 1
    size = idx
    H = np.zeros((size, size), dtype=np.int32)
    
    for u in range(n):
        for v in adj[u]:
            u_v = state_to_idx[(u, v)]
            for w in adj[v]:
                if w != u:
                    v_w = state_to_idx[(v, w)]
                    H[u_v, v_w] = 1
                    
    H_power = np.eye(size, dtype=np.int32)
    for L in range(2, g):
        H_power = H_power @ H
        if np.any(np.diag(H_power) > 0):
            return False
    return True

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
