import time
import math


def construct():
    t0 = time.time()
    DEADLINE = 55.0
    N = 256

    def is_sq(x):
        r = math.isqrt(x)
        return r * r == x

    # Conflict adjacency (symmetric), 0-indexed over [0, N-1]
    adj = [0] * N
    for a in range(1, N + 1):
        ia = a - 1
        for b in range(a, N + 1):
            if is_sq(a + b):
                ib = b - 1
                adj[ia] |= (1 << ib)
                adj[ib] |= (1 << ia)

    # Self-conflict: 2a is a perfect square -> a cannot be chosen
    selfconf = [False] * N
    for a in range(1, N + 1):
        if is_sq(2 * a):
            selfconf[a - 1] = True

    # Universe of selectable vertices
    uni = 0
    for i in range(N):
        if not selfconf[i]:
            uni |= (1 << i)

    # Complement adjacency (within universe): non-conflict pairs
    comp = [0] * N
    for i in range(N):
        if not ((uni >> i) & 1):
            comp[i] = 0
        else:
            comp[i] = uni & ~(adj[i] | (1 << i))

    cdeg = [comp[i].bit_count() for i in range(N)]

    best_size = 0
    best_mask = 0

    def color_sort(P):
        verts = []
        c = P
        while c:
            lsb = c & (-c)
            i = lsb.bit_length() - 1
            verts.append(i)
            c ^= lsb
        verts.sort(key=lambda i: cdeg[i])
        color_classes = []
        colors = {}
        order = []
        for i in verts:
            bit = 1 << i
            placed = False
            for ci, cb in enumerate(color_classes):
                if (comp[i] & cb) == 0:
                    color_classes[ci] = cb | bit
                    colors[i] = ci + 1
                    order.append(i)
                    placed = True
                    break
            if not placed:
                color_classes.append(bit)
                colors[i] = len(color_classes)
                order.append(i)
        return order, colors

    def expand(P, size, chosen):
        nonlocal best_size, best_mask
        if time.time() > DEADLINE:
            return
        if P == 0:
            if size > best_size:
                best_size = size
                best_mask = chosen
            return
        order, colors = color_sort(P)
        for idx in range(len(order) - 1, -1, -1):
            if time.time() > DEADLINE:
                return
            i = order[idx]
            if size + colors[i] <= best_size:
                return
            bit = 1 << i
            expand(P & comp[i], size + 1, chosen | bit)
            P ^= bit

    def greedy_clique():
        P = uni
        chosen = 0
        size = 0
        while P:
            best = None
            bd = -1
            c = P
            while c:
                lsb = c & (-c)
                i = lsb.bit_length() - 1
                d = (comp[i] & P).bit_count()
                if d > bd:
                    bd = d
                    best = i
                c ^= lsb
            chosen |= (1 << best)
            size += 1
            P &= comp[best]
        return size, chosen

    gs, gm = greedy_clique()
    best_size = gs
    best_mask = gm

    expand(uni, 0, 0)

    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result.append(i + 1)
        m ^= lsb
    if not result:
        result = [1]
    return result
