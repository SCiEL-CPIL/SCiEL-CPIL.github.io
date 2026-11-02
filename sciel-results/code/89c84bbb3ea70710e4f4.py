import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Try Cayley graph on Z_16 x Z_20 with fast girth check
        result = _try_cayley_320(deadline)
        if result is not None:
            return result

        # Fallback: LUW Construction
        return _luw_construct(k, g)

    # Fallback: LUW Construction
    return _luw_construct(k, g)


def _try_cayley_320(deadline):
    """
    Search for a 4-regular Cayley graph on Z_16 x Z_20 with girth >= 10.
    Generators: (1,0), (-1,0), (a,b), (-a,-b)
    Girth check: no non-trivial word of length < 10 in generators equals identity.
    For abelian group, this means: no (i,j) with |i|+|j| < 10 such that
    i*(1,0) + j*(a,b) = (0,0) mod (16,20), except i=j=0.
    """
    m, n = 16, 20
    total = m * n

    # Precompute: for each (a,b), check if there's a short relation
    # i*(1,0) + j*(a,b) = (i + j*a, j*b) = (0,0) mod (m,n)
    # So: i ≡ -j*a (mod m) and j*b ≡ 0 (mod n)
    # For j=0: i ≡ 0 (mod m), so i=0 (since |i| < 10 < 16) -> trivial
    # For j≠0: j*b ≡ 0 (mod n) means n | j*b
    #   And i = -j*a mod m, with |i| < 10, |j| < 10, |i|+|j| < 10

    def has_short_relation(a, b):
        """Check if there's a non-trivial (i,j) with |i|+|j| < 10 and |i|+|j| >= 1
        such that i*(1,0) + j*(a,b) = (0,0) mod (m,n)."""
        for j in range(-9, 10):
            if j == 0:
                # i*(1,0) = (0,0) mod m => i ≡ 0 mod 16, |i|<10 => i=0, trivial
                continue
            # Check j*b ≡ 0 mod n
            if (j * b) % n != 0:
                continue
            # i ≡ -j*a mod m
            i_mod = (-j * a) % m
            # Find smallest |i| such that i ≡ i_mod (mod m)
            # i can be i_mod or i_mod - m or i_mod + m, etc.
            # We need |i| < 10 - |j| (since |i|+|j| < 10)
            max_i = 10 - abs(j) - 1  # |i| + |j| <= 9, so |i| <= 9 - |j|
            if max_i < 0:
                continue
            # Check candidates for i
            candidates = []
            # i = i_mod
            if abs(i_mod) <= max_i:
                candidates.append(i_mod)
            # i = i_mod - m
            if abs(i_mod - m) <= max_i:
                candidates.append(i_mod - m)
            # i = i_mod + m
            if abs(i_mod + m) <= max_i:
                candidates.append(i_mod + m)
            # Also check negative representations
            # i_mod is in [0, m-1], so i_mod - m is in [-m, -1]
            # We already covered i_mod and i_mod-m. Let's also check i_mod + m if small.
            for i in candidates:
                if i == 0 and j == 0:
                    continue  # trivial
                if abs(i) + abs(j) < 10 and abs(i) + abs(j) >= 1:
                    # Verify: i*(1,0) + j*(a,b) = (i + j*a, j*b)
                    if (i + j * a) % m == 0 and (j * b) % n == 0:
                        return True
        return False

    def verify_girth_bfs(a, b):
        """Full BFS verification of girth >= 10 from (0,0)."""
        gens = [
            (1, 0),
            (m - 1, 0),
            (a, b),
            ((-a) % m, (-b) % n)
        ]

        # BFS from (0,0), tracking depth
        dist = {}
        dist[(0, 0)] = 0
        queue = deque([(0, 0)])

        while queue:
            u, v = queue.popleft()
            d = dist[(u, v)]
            if d >= 5:  # We only need to check up to depth 4 for girth 10
                continue
            for g1, g2 in gens:
                nu = (u + g1) % m
                nv = (v + g2) % n
                if (nu, nv) not in dist:
                    dist[(nu, nv)] = d + 1
                    queue.append((nu, nv))
                else:
                    # Found a cycle of length d + 1 + dist[(nu,nv)]
                    # But this is the cycle through the edge (u,v)->(nu,nv)
                    # The actual cycle length is d + 1 + dist[(nu,nv)] only if
                    # the shortest path from (nu,nv) to (0,0) doesn't go through (u,v)
                    # This is a conservative check - if d+1+dist[(nu,nv)] < 10,
                    # there IS a cycle of that length or shorter.
                    cycle_len = d + 1 + dist[(nu, nv)]
                    if cycle_len < 10:
                        return False
        return True

    # Search over all (a, b)
    for a in range(m):
        if time.time() > deadline:
            return None
        for b in range(n):
            if time.time() > deadline:
                return None

            # Skip trivial/generator overlaps
            if a == 0 and b == 0:
                continue
            if a == 1 and b == 0:
                continue
            if a == m - 1 and b == 0:
                continue

            # Check that (a,b) is not the inverse of (1,0) or (-1,0)
            # (a,b) != (1,0) and (a,b) != (m-1,0) already checked

            # Check that (a,b) != (0,0) already checked

            # Check that 2*(a,b) != (0,0) to avoid order-2 issues
            # Actually, order 2 is fine, but let's check the generators are distinct
            # gens: (1,0), (m-1,0), (a,b), (-a,-b)
            # Need (a,b) != (1,0), (a,b) != (m-1,0), (a,b) != (-a,-b) mod (m,n)
            # (a,b) != (-a,-b) means 2a != 0 mod m and 2b != 0 mod n
            if (2 * a) % m == 0 and (2 * b) % n == 0:
                continue

            # Also need (a,b) != (1,0) and (a,b) != (m-1,0) - already checked
            # And (-a,-b) != (1,0) and (-a,-b) != (m-1,0)
            if (-a) % m == 1 and (-b) % n == 0:
                continue
            if (-a) % m == m - 1 and (-b) % n == 0:
                continue

            # Fast algebraic check
            if has_short_relation(a, b):
                continue

            # Full BFS verification
            if verify_girth_bfs(a, b):
                # Build edges
                edges = []
                for u in range(m):
                    for v in range(n):
                        # Edge (u,v) -- (u+1, v)
                        nu = (u + 1) % m
                        nv = v
                        if (u, v) < (nu, nv):
                            edges.append((u * n + v, nu * n + nv))

                        # Edge (u,v) -- (u+a, v+b)
                        nu = (u + a) % m
                        nv = (v + b) % n
                        if (u, v) < (nu, nv):
                            edges.append((u * n + v, nu * n + nv))

                # Verify degree
                deg = [0] * total
                for e1, e2 in edges:
                    deg[e1] += 1
                    deg[e2] += 1
                if all(d == 4 for d in deg):
                    return total, edges

    return None


def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
