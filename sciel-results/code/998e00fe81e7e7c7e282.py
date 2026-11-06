import time
import random

BASE = 51975
# all divisors > 1
def _divisors(n):
    ds = []
    for d in range(2, n+1):
        if n % d == 0:
            ds.append(d)
    return ds

MODULI = _divisors(BASE)
# positions for each modulus and residue: precompute index lists
# For speed, for each m, positions[r] = list of indices
_POS = {}
for m in MODULI:
    arr = []
    for r in range(m):
        arr.append(range(r, BASE, m))
    _POS[m] = arr

def _uncovered_count(cover):
    # cover: array of uint8 counts
    return int((cover == 0).sum())

def _build_cover(classes):
    cover = bytearray(BASE)  # counts, but we may exceed 255? max 47, fine
    for m, r in classes:
        for idx in _POS[m][r]:
            cover[idx] += 1
    return cover

def _greedy_init():
    # process moduli in increasing order, pick r maximizing new coverage
    cover = bytearray(BASE)
    classes = []
    for m in sorted(MODULI):
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for idx in _POS[m][r]:
                if cover[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        for idx in _POS[m][best_r]:
            cover[idx] += 1
    return classes, cover

def _random_init(rng):
    classes = []
    cover = bytearray(BASE)
    for m in MODULI:
        r = rng.randrange(m)
        classes.append([m, r])
        for idx in _POS[m][r]:
            cover[idx] += 1
    return classes, cover

def _local_search(classes, cover, deadline, rng):
    # hill climbing with first-improvement / best-improvement per modulus
    cur = _uncovered_count(cover)
    order = list(range(len(MODULI)))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # randomize order occasionally to escape
        if rng.random() < 0.5:
            rng.shuffle(order)
        for i in order:
            m = MODULI[i]
            r_old = classes[i][1]
            best_gain = 0  # we want to reduce uncovered, so gain = reduction
            best_r = r_old
            # compute for each candidate r the reduction in uncovered
            # reduction = (# in B with count 0) - (# in A with count 1)
            A = _POS[m][r_old]
            for r in range(m):
                if r == r_old:
                    continue
                B = _POS[m][r]
                red = 0
                for idx in B:
                    if cover[idx] == 0:
                        red += 1
                for idx in A:
                    if cover[idx] == 1:
                        red -= 1
                if red > best_gain:
                    best_gain = red
                    best_r = r
            if best_r != r_old and best_gain > 0:
                # apply
                for idx in A:
                    cover[idx] -= 1
                for idx in _POS[m][best_r]:
                    cover[idx] += 1
                classes[i][1] = best_r
                cur -= best_gain
                improved = True
    return classes, cover, cur

def construct():
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(12345)

    # best solution
    best_classes, best_cover, best_score = None, None, None

    def _consider(classes, cover):
        nonlocal best_classes, best_cover, best_score
        s = _uncovered_count(cover)
        if best_score is None or s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
            best_cover = bytearray(cover)

    # greedy init + local search
    g_classes, g_cover = _greedy_init()
    g_classes, g_cover, g_score = _local_search(g_classes, g_cover, deadline, rng)
    _consider(g_classes, g_cover)

    # random restarts
    while time.time() < deadline:
        r_classes, r_cover = _random_init(rng)
        r_classes, r_cover, r_score = _local_search(r_classes, r_cover, deadline, rng)
        _consider(r_classes, r_cover)

    return best_classes
