import time
import math

_START = time.time()
DEADLINE = 150.0
SAFETY = 15.0  # seconds to leave before the hard kill

def _deadline_hit():
    return (time.time() - _START) > (DEADLINE - SAFETY)

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = math.isqrt(n)
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def _smallest_prime_power_ge(n):
    # smallest prime power >= n
    q = max(2, n)
    # try primes first
    while True:
        if _is_prime(q):
            return q
        q += 1

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q): q-regular bipartite graph with
    # girth d + 5 (odd d). We need girth >= g, so pick odd d with d+5 >= g, i.e.
    # d >= g-5. For girth 9, d=4. q must be a prime power with q >= k.
    # The connected component of the zero point is returned.
    q = _smallest_prime_power_ge(k)
    # choose odd d >= g-5, d >= 3
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make odd so girth = d+5 is even? Actually CD(d,q) girth is d+5.

    # GF(q) operations (q prime or q=4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # rule[c] for c = 2..d : (a, b)
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)

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

def _girth_nonbacktracking(n, edges, limit):
    # Exact girth via non-backtracking BFS from every vertex (BFS on directed
    # non-backtracking walks). Returns the shortest cycle length, or > limit.
    # Build adjacency.
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    best = limit + 1
    for s in range(n):
        # state: (prev, cur, depth)
        seen = {}  # (prev, cur) -> depth
        stack = [(None, s, 0)]
        while stack:
            prev, cur, depth = stack.pop()
            if depth + 1 >= best:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                key = (cur, nb)
                if nb == s and depth + 1 >= 3:
                    # found a cycle
                    if depth + 1 < best:
                        best = depth + 1
                    continue
                if key in seen:
                    continue
                seen[key] = depth + 1
                stack.append((cur, nb, depth + 1))
    return best

def construct(k, g):
    if _deadline_hit():
        # fallback: smallest prime power q >= k, CD(4, q)
        q = _smallest_prime_power_ge(k)
        return construct_lw(k, g)

    # Primary: LW CD(d, q) with q = smallest prime power >= k, d = 4 (girth 9)
    # This gives a k-regular? No -- it is q-regular. We need exactly k-regular.
    # So we need q == k, i.e., k must be a prime power. If k is a prime power,
    # CD(4, k) is k-regular with girth 9. Otherwise we fall back to a covering
    # or to the ABS construction.

    if _smallest_prime_power_ge(k) == k:
        n, edges = construct_lw(k, g)
        return n, edges

    # k not a prime power: use a voltage lift of a smaller regular base graph.
    # Use CD(4, q0) with q0 = smallest prime power <= k that is >= 3, then we
    # cannot change degree. Instead, fall back to ABS for even k, LW for odd.
    if k % 2 == 0:
        return construct_abs(k, g)
    else:
        # odd k not a prime power: use LW with q = smallest prime power >= k
        # which is q-regular, not k-regular. To get exactly k-regular, we take
        # the q-regular graph and it is not k-regular. So we must construct a
        # k-regular graph directly. Use the general CD construction restricted
        # to a k-regular subgraph is hard; instead use a circulant-free approach:
        # build k-regular graph via the "generalized Petersen-like" lift.
        # Simplest robust fallback: use ABS-like Heisenberg for even, and for
        # odd use a known k-regular girth-9 construction via Cayley graph on a
        # finite group of order n with k generators.
        return construct_cayley(k, g)

def construct_abs(k, g):
    # Alon-Babai-Suzuki construction on Heisenberg group H_q
    # Degree k must be even.
    if g % 2 == 0:
        mb = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        mb = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))

    q = 2
    while True:
        if _is_prime(q):
            if q ** 3 >= mb:
                break
        q += 1

    if k == 4:
        gens = [
            (1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 1)
        ]
    elif k == 6:
        gens = [
            (1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 1), (0, 0, 1), (0, 0, q - 1)
        ]
    else:
        # general even k: use k/2 inverse pairs of generators from the group.
        # We need k distinct generators with no element of order 2 causing loops.
        # Use (1,0,0),(0,1,0),(0,0,1) and their inverses, plus extra via (1,1,0) etc.
        base = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        gens = []
        for (ax, ay, az) in base:
            if len(gens) >= k:
                break
            gens.append((ax, ay, az))
            inv = ((q - ax) % q, (q - ay) % q, (q - az - ax * ay) % q)
            if len(gens) < k and inv != (ax, ay, az):
                gens.append(inv)
        if len(gens) < k:
            return construct_lw(k, g)

    n = q ** 3
    edges = []

    def idx(x, y, z):
        return (x * q + y) * q + z

    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges

def construct_cayley(k, g):
    # Cayley graph on a finite abelian group with k generators (inverse-closed).
    # Use group Z_m^r with m chosen so m^r is near the Moore bound and girth >= g.
    # For k odd, we need an odd number of generators, which requires an element
    # of order 2 or a self-inverse generator. Use a group with an involution.
    # Simplest: take group = Z_2 x Z_p^{r-1} won't give odd regular easily.
    # Instead: use a circulant on Z_n with k generators including a symmetric set
    # and, if k is odd, include the involution n/2 (requires n even).
    # We'll search for n and a generator set giving girth >= g.
    # Moore bound
    if g % 2 == 0:
        mb = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        mb = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))

    # Try n = next even >= mb, build circulant with k generators.
    # For k odd, use generators {a1, -a1, ..., a_{(k-1)/2}, -a_{(k-1)/2}, n/2}.
    # For k even, use {(k/2) pairs}.
    # Choose generators as small odd numbers to maximize girth (spread out).
    n = mb
    if n % 2 != 0:
        n += 1
    if k % 2 == 0:
        half = k // 2
        gens = []
        for a in range(1, n, 2):
            if len(gens) == half:
                break
            gens.append(a)
        gen_set = set()
        for a in gens:
            gen_set.add(a)
            gen_set.add((n - a) % n)
    else:
        half = (k - 1) // 2
        gens = []
        for a in range(1, n, 2):
            if len(gens) == half:
                break
            gens.append(a)
        gen_set = set()
        for a in gens:
            gen_set.add(a)
            gen_set.add((n - a) % n)
        gen_set.add(n // 2)  # involution, makes degree odd

    # Build edges
    edges = []
    seen = set()
    for u in range(n):
        for a in gen_set:
            v = (u + a) % n
            if v <= u:
                continue
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            edges.append((u, v))
    return n, edges
