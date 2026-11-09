import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    N = 256
    masks = [0] * (N + 1)
    deg = [0] * (N + 1)
    for a in range(1, N + 1):
        a2 = a * a
        m = 0
        for b in range(1, N + 1):
            if b != a and a2 % (a + b) == 0:
                m |= (1 << (b - 1))
        masks[a] = m
        deg[a] = bin(m).count('1')

    def popcount(x):
        return bin(x).count('1')

    best_mask = 0
    best_size = 0

    for _ in range(20000):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        order.sort(key=lambda v: (deg[v], random.random()))
        chosen = 0
        for v in order:
            if (masks[v] & chosen) == 0:
                chosen |= (1 << (v - 1))
        sz = popcount(chosen)
        if sz > best_size:
            best_size = sz
            best_mask = chosen

    cur = best_mask
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if (cur & bit) == 0 and (masks[v] & cur) == 0:
                cur |= bit
                improved = True
        if improved:
            continue
        cur_list = [v for v in range(1, N + 1) if (cur >> (v - 1)) & 1]
        random.shuffle(cur_list)
        for u in cur_list:
            if time.time() > deadline:
                break
            new_cur = cur & ~(1 << (u - 1))
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if (new_cur & bit) == 0 and (masks[v] & new_cur) == 0:
                    new_cur |= bit
            if popcount(new_cur) > popcount(cur):
                cur = new_cur
                improved = True
                break

    result = [v for v in range(1, N + 1) if (cur >> (v - 1)) & 1]
    if not result:
        return [1]
    return result
