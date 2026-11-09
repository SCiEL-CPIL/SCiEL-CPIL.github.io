import time
from itertools import combinations

_DEADLINE = time.time() + 115.0  # safety margin vs 120s wall-clock

def _init(n, seed):
    import random
    r = random.Random(seed)
    ntri = n * (n - 1) * (n - 2) // 6
    colors = bytearray(r.getrandbits(8) & 1 for _ in range(ntri))
    red = [0] * n
    blue = [0] * n
    t = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if colors[t]:
                    blue[a] |= (1 << (b * n + c))
                    blue[b] |= (1 << (a * n + c))
                    blue[c] |= (1 << (a * n + b))
                else:
                    red[a] |= (1 << (b * n + c))
                    red[b] |= (1 << (a * n + c))
                    red[c] |= (1 << (a * n + b))
                t += 1
    return colors, red, blue

def _total_violations(red, blue, n):
    return sum((red[v].bit_count() >= 3) for v in range(n)) + \
           sum((blue[v].bit_count() >= 4) for v in range(n))

def _neighbors(mask, n):
    out = []
    m = mask
    while m:
        lsb = m & (-m)
        pos = lsb.bit_length() - 1
        out.append(pos // n)
        m ^= lsb
    return out

def _solve(n, seed, step_budget):
    import random
    colors, red, blue = _init(n, seed)
    idx = {}
    t = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                idx[(a, b, c)] = t
                t += 1
    r = random.Random(seed + 7919)
    steps = 0
    best = _total_violations(red, blue, n)
    while best > 0 and steps < step_budget and time.time() < _DEADLINE:
        # find violating vertices
        red_v = [v for v in range(n) if red[v].bit_count() >= 3]
        blue_v = [v for v in range(n) if blue[v].bit_count() >= 4]
        if not red_v and not blue_v:
            break
        # prefer the more "excessive" violation type
        if red_v and (not blue_v or len(red_v) >= len(blue_v)):
            v = r.choice(red_v)
            kind = 0
        else:
            v = r.choice(blue_v)
            kind = 1
        if kind == 0:
            nbrs = _neighbors(red[v], n)
            if len(nbrs) < 2:
                steps += 1
                continue
            i = r.randrange(len(nbrs))
            j = r.randrange(len(nbrs) - 1)
            if j >= i:
                j += 1
            x, y = nbrs[i], nbrs[j]
            if x == y:
                steps += 1
                continue
            a, b, c = sorted((v, x, y))
            tri = idx[(a, b, c)]
            if colors[tri] != 0:
                steps += 1
                continue
            colors[tri] = 1
            for (p, q, rr) in ((a, b, c), (b, a, c), (c, a, b)):
                red[p] &= ~(1 << (min(q, rr) * n + max(q, rr)))
                blue[p] |= (1 << (min(q, rr) * n + max(q, rr)))
        else:
            nbrs = _neighbors(blue[v], n)
            if len(nbrs) < 2:
                steps += 1
                continue
            i = r.randrange(len(nbrs))
            j = r.randrange(len(nbrs) - 1)
            if j >= i:
                j += 1
            x, y = nbrs[i], nbrs[j]
            if x == y:
                steps += 1
                continue
            a, b, c = sorted((v, x, y))
            tri = idx[(a, b, c)]
            if colors[tri] != 1:
                steps += 1
                continue
            colors[tri] = 0
            for (p, q, rr) in ((a, b, c), (b, a, c), (c, a, b)):
                blue[p] &= ~(1 << (min(q, rr) * n + max(q, rr)))
                red[p] |= (1 << (min(q, rr) * n + max(q, rr)))
        steps += 1
        best = _total_violations(red, blue, n)
    return best, colors

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    best_n = 0
    best_colors = None
    for n in range(9, 41):
        if time.time() > _DEADLINE - 5:
            break
        step_budget = max(5000, 60000 - (n - 9) * 1500)
        for seed in range(0, 40):
            if time.time() > _DEADLINE - 5:
                break
            best, colors = _solve(n, seed, step_budget)
            if best == 0:
                best_n = n
                best_colors = list(colors)
                break
    if best_colors is None:
        return {'n': 1, 'colors': []}
    return {'n': best_n, 'colors': best_colors}
