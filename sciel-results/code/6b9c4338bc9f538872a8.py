import time
import random

_TARGET = 200
_MASK_ALL = (1 << 100) - 1
_LIMIT = (1 << (_TARGET + 1)) - 1

def _safe(mask):
    s = 1
    m = mask
    while m:
        lb = m & -m
        i = lb.bit_length() - 1
        s |= s << (i + 1)
        s &= _LIMIT
        m ^= lb
    return ((s >> _TARGET) & 1) == 0

def construct():
    t0 = time.time()
    DEAD = t0 + 52.0
    rng = random.Random(12345)

    # initial [66,100]
    mask = 0
    for x in range(66, 101):
        mask |= 1 << (x - 1)

    best_mask = mask
    best_size = mask.bit_count()

    def consider(m):
        nonlocal best_mask, best_size
        sz = m.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = m

    def grow(m):
        # greedy add
        while time.time() < DEAD:
            added = False
            for x in range(1, 101):
                b = 1 << (x - 1)
                if m & b:
                    continue
                if _safe(m | b):
                    m |= b
                    added = True
                    consider(m)
            if not added:
                break
        return m

    mask = grow(mask)

    def elems(m):
        return [x for x in range(1, 101) if (m >> (x - 1)) & 1]
    def non(m):
        return [x for x in range(1, 101) if not ((m >> (x - 1)) & 1)]

    # local search: remove 1 add 2
    def local12(m):
        while time.time() < DEAD:
            improved = False
            es = elems(m); ns = non(m)
            for r in es:
                rb = 1 << (r - 1)
                base = m & ~rb
                for i in range(len(ns)):
                    a = ns[i]; ab = 1 << (a - 1)
                    for j in range(i + 1, len(ns)):
                        b = ns[j]; bb = 1 << (b - 1)
                        nm = base | ab | bb
                        if _safe(nm):
                            m = nm
                            improved = True
                            consider(m)
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return m

    mask = local12(mask)

    # targeted exhaustive: [66,100]\{z} + {a,b}
    base66 = 0
    for x in range(66, 101):
        base66 |= 1 << (x - 1)
    small = [x for x in range(1, 66)]
    found36 = None
    for z in range(66, 101):
        if time.time() > DEAD:
            break
        zb = 1 << (z - 1)
        base = base66 & ~zb
        for i in range(len(small)):
            a = small[i]; ab = 1 << (a - 1)
            for j in range(i + 1, len(small)):
                b = small[j]; bb = 1 << (b - 1)
                nm = base | ab | bb
                if _safe(nm):
                    found36 = nm
                    break
            if found36:
                break
        if found36:
            break
    if found36 is not None:
        consider(found36)
        mask = found36
        mask = grow(mask)
        mask = local12(mask)

    # random restarts: random 35-sets biased large
    while time.time() < DEAD - 5:
        m = 0
        for x in range(70, 101):
            m |= 1 << (x - 1)
        extra = rng.sample(range(1, 70), 4)
        for x in extra:
            m |= 1 << (x - 1)
        m = grow(m)
        m = local12(m)
        if m.bit_count() > best_size:
            mask = m

    res = [i + 1 for i in range(100) if (best_mask >> i) & 1]
    return res
