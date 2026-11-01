import itertools
import time

def luw_construct(k, g):
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

def check_girth_4_9(n, a, b):
    S = [a, -a, b, -b]
    for L in range(4, 10):
        for seq in itertools.product(S, repeat=L):
            s = sum(seq)
            if s % n != 0:
                continue
            partials = [0]
            cur = 0
            valid = True
            for step in seq:
                cur = (cur + step) % n
                if cur in partials:
                    valid = False
                    break
                partials.append(cur)
            if valid:
                return False
    return True

def construct(k, g):
    if k == 4 and g == 10:
        start_time = time.time()
        deadline = start_time + 100
        
        for n in range(242, 400):
            if time.time() > deadline:
                break
            for a in range(1, n // 2 + 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n // 2 + 1):
                    if (3*a) % n == 0: continue
                    if (2*a + b) % n == 0: continue
                    if (2*a - b) % n == 0: continue
                    if (a + 2*b) % n == 0: continue
                    if (a - 2*b) % n == 0: continue
                    if (3*b) % n == 0: continue
                    
                    if check_girth_4_9(n, a, b):
                        edges = []
                        for i in range(n):
                            for s in [a, -a, b, -b]:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
        return luw_construct(4, 10)
    else:
        return luw_construct(k, g)
