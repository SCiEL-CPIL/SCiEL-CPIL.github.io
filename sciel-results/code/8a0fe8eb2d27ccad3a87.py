"""LUW graph construction with correct girth parameter d = (g-2)//2."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110
    q = k
    d = (g - 2) // 2
    if d < 2:
        d = 2

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    n_points = q ** d
    total = 2 * n_points
    qd = n_points

    edges = set()

    for p_idx in range(n_points):
        coords = [0] * d
        tmp = p_idx
        for i in range(d - 1, -1, -1):
            coords[i] = tmp % q
            tmp //= q

        for l1 in range(q):
            l = [0] * d
            l[0] = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                l[c - 1] = add(coords[c - 1], mul(l[a - 1], coords[b - 1]))
            l_idx = qd + 0
            # Compute line index
            li = 0
            for i in range(d):
                li = li * q + l[i]
            l_idx = qd + li
            edges.add((p_idx, l_idx))

    return total, list(edges)
# EVOLVE-BLOCK-END
