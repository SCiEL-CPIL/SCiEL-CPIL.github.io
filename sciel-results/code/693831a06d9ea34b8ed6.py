import time
import random
from collections import deque


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    if k == 4 and g == 10:
        res = _voltage_lift_search(start_time, deadline, g)
        if res is not None:
            return res
        return _luw_construct(k, g)

    return _luw_construct(k, g)


def _voltage_lift_search(start_time, deadline, g):
    M = 32
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))

    badj = [[] for _ in range(10)]
    for u, v in base_edges:
        badj[u].append(v)
        badj[v].append(u)

    def build(v):
        edges = []
        for idx in range(len(base_edges)):
            u, vbase = base_edges[idx]
            w = v[u][vbase - 5]
            base_u = u * M
            base_v = vbase * M
            if w == 0:
                for x in range(M):
                    edges.append((base_u + x, base_v + x))
            else:
                for x in range(M):
                    y = x + w
                    if y >= M:
                        y -= M
                    edges.append((base_u + x, base_v + y))
        return edges

    def girth_ok(v, min_girth):
        max_depth = min_girth // 2
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = [(s, -1, 0)]
            qi = 0
            while qi < len(q):
                u, p, d = q[qi]
                qi += 1
                if d >= max_depth:
                    continue
                for vv in badj[u]:
                    if vv == p:
                        continue
                    key = (vv, u)
                    if key in seen:
                        if d + 1 < min_girth:
                            return False
                    else:
                        seen.add(key)
                        q.append((vv, u, d + 1))
        return True

    # Try 3-parameter form: v[i][j] = (a*i + b*j + d) % M
    for a in range(M):
        if time.time() > deadline - 5.0:
            break
        for b in range(M):
            if time.time() > deadline - 5.0:
                break
            for d in range(M):
                v = [[((a * i + b * j + d) % M) for j in range(5)] for i in range(5)]
                if girth_ok(v, g):
                    return 320, build(v)

    # Try 4-parameter form: v[i][j] = (a*i + b*j + c*i*j + d) % M for small c
    for c in range(1, 8):
        if time.time() > deadline - 5.0:
            break
        for a in range(M):
            if time.time() > deadline - 5.0:
                break
            for b in range(M):
                if time.time() > deadline - 5.0:
                    break
                for d in range(M):
                    v = [[((a * i + b * j + c * i * j + d) % M) for j in range(5)] for i in range(5)]
                    if girth_ok(v, g):
                        return 320, build(v)

    # Random search with local search
    def random_v():
        return [[random.randint(0, M - 1) for _ in range(5)] for _ in range(5)]

    def count_short_cycles(v, min_girth):
        max_depth = min_girth // 2
        count = 0
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = [(s, -1, 0)]
            qi = 0
            while qi < len(q):
                u, p, d = q[qi]
                qi += 1
                if d >= max_depth:
                    continue
                for vv in badj[u]:
                    if vv == p:
                        continue
                    key = (vv, u)
                    if key in seen:
                        count += 1
                    else:
                        seen.add(key)
                        q.append((vv, u, d + 1))
        return count

    best_count = float('inf')
    best_v = None
    while time.time() < deadline - 5.0:
        v = random_v()
        cur_count = count_short_cycles(v, g)
        if cur_count == 0:
            return 320, build(v)
        if cur_count < best_count:
            best_count = cur_count
            best_v = [row[:] for row in v]

        cur_v = [row[:] for row in v]
        cur_c = cur_count
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            i = random.randrange(5)
            j = random.randrange(5)
            old = cur_v[i][j]
            new = random.randint(0, M - 1)
            if new == old:
                continue
            cur_v[i][j] = new
            new_c = count_short_cycles(cur_v, g)
            if new_c == 0:
                return 320, build(cur_v)
            if new_c < cur_c or random.random() < 0.1:
                cur_c = new_c
                if cur_c < best_count:
                    best_count = cur_c
                    best_v = [row[:] for row in cur_v]
            else:
                cur_v[i][j] = old

    if best_v is not None and best_count == 0:
        return 320, build(best_v)
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
        r = (c - 5) % 4
        c0 = c - r
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
            yield (1 - side, tuple(y))

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
