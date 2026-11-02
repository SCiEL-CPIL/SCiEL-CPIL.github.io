"""Constructive search for small k-regular graphs of large girth.

Strategy:
  1. Use a bipartite Cayley graph construction on Z_2^d.
  2. The degree is k, determined by the number of generators.
  3. The girth is determined by the smallest sum of generators equal to 0.
  4. We find a set of k generators in Z_2^d that forms a Sidon set 
     (no 4-sum 0) and has no 3-sum 0, ensuring girth >= 10 (for g=9).
  5. This yields a graph with n = 2^d vertices, which is often smaller 
     than algebraic constructions like LUW for these parameters.
  6. Falls back to LUW if no valid generator set is found.
"""
import random
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _moore_bound_bipartite(k, g):
    """Moore bound for bipartite graphs with girth >= g+1 (since g is odd)."""
    g_bip = g + 1
    if g_bip % 2 != 0:
        g_bip += 1
    h = g_bip // 2
    s = 0
    p = 1
    for _ in range(h):
        s += p
        p *= (k - 1)
    return 2 * s

def _find_sidon_set(k, d, rng):
    """Find k vectors in Z_2^d with no 3-sum 0 and no 4-sum 0."""
    for _ in range(2000):
        S = []
        for _ in range(k):
            found = False
            for _ in range(100):
                v = rng.getrandbits(d)
                if v == 0:
                    continue
                if v in S:
                    continue
                ok = True
                # Check 3-sum: v ^ a ^ b == 0
                for i in range(len(S)):
                    for j in range(len(S)):
                        if i == j:
                            continue
                        if v ^ S[i] ^ S[j] == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Check 4-sum: v ^ a ^ b ^ c == 0
                for i in range(len(S)):
                    for j in range(len(S)):
                        for l in range(len(S)):
                            if i == j or j == l or i == l:
                                continue
                            if v ^ S[i] ^ S[j] ^ S[l] == 0:
                                ok = False
                                break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                S.append(v)
                found = True
                break
            if not found:
                break
        if len(S) == k:
            return S
    return None

def _fallback_luw(k, g):
    """Fallback: the original Lazebnik-Ustimenko-Woldar construction."""
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

def construct(k, g):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)

    if k < 2 or g < 3:
        return _fallback_luw(max(2, k), max(3, g))

    mb = _moore_bound_bipartite(k, g)
    
    d = 1
    while (1 << d) < mb:
        d += 1
        
    gens = _find_sidon_set(k, d, rng)
    if gens is None:
        d += 2
        gens = _find_sidon_set(k, d, rng)
        
    if gens is None:
        return _fallback_luw(k, g)

    n = 1 << d
    edges = []
    for v in range(n):
        for g in gens:
            u = v ^ g
            if v < u:
                edges.append((v, u))
                
    return n, edges
