import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115
    assert k == 4, "Specialized for k=4"

    def check_girth(n, a, b, min_girth):
        """Return True if 4-regular circulant C(n; a, b) has girth >= min_girth."""
        # 4-cycles
        if (2 * a) % n == 0 or (2 * b) % n == 0:
            return False
        if (a + b) % n == 0 or (a - b) % n == 0:
            return False
        # 6-cycles and 8-cycles: check all (x,y) with |x|+|y| = L for L in 6, 8
        for L in range(4, min_girth):
            for x in range(1, L):
                y = L - x
                for sx in (1, -1):
                    for sy in (1, -1):
                        if (sx * x * a + sy * y * b) % n == 0:
                            return False
        return True

    best_n = None
    best_ab = None

    for n in range(243, 360):
        if time.time() > deadline:
            break
        if best_n is not None and n >= best_n:
            break
        for a in range(1, n // 2 + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, n // 2 + 1):
                if check_girth(n, a, b, g):
                    best_n = n
                    best_ab = (a, b)
                    break
            if best_n is not None:
                break
        if best_n is not None:
            break

    if best_n is not None:
        a, b = best_ab
        n = best_n
        edges = []
        for i in range(n):
            for d in (a, b):
                j = (i + d) % n
                if i < j:
                    edges.append((i, j))
                else:
                    edges.append((j, i))
        edges = list(set(edges))
        return n, edges
    else:
        # Fallback: Lazebnik-Ustimenko-Woldar construction
        q = 4
        d = max(3, g - 5)
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            if r == 0: rule[c] = (1, c0-2)
            elif r == 1: rule[c] = (c0-1, 1)
            elif r == 2: rule[c] = (c0, 1)
            else: rule[c] = (1, c0+1)
        start = (0, (0,)+(0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vi in range(len(order)):
            vertex = order[vi]
            side, x = vertex
            for first in range(q):
                y = [0, first] + [0]*(d-1)
                for c in range(2, d+1):
                    ar, br = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[ar], x[br]))
                    else:
                        y[c] = sub(x[c], mul(x[ar], y[br]))
                other = (1-side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
# EVOLVE-BLOCK-END
