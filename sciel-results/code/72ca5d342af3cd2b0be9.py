import random
import time

DEADLINE = 140.0  # safety margin under 150s

def _superset_mark(triples, n=16):
    """Return boolean list has_triple[S] for S in 0..2^n-1, and alpha (max independent size)."""
    size = 1 << n
    has = bytearray(size)  # 0/1
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        # iterate all supersets of mask
        rest = ((1 << n) - 1) ^ mask
        sub = rest
        while True:
            has[mask | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & rest
    # find largest S with has[S]==0
    best = 0
    for S in range(size):
        if has[S] == 0:
            c = S.bit_count()
            if c > best:
                best = c
    return best

def _is_linear(triples):
    # check any two share at most one point
    m = len(triples)
    for i in range(m):
        a, b, c = triples[i]
        for j in range(i + 1, m):
            x, y, z = triples[j]
            inter = len({a, b, c} & {x, y, z})
            if inter > 1:
                return False
    return True

def _random_linear_system(rng, n=16, max_triples=40):
    triples = []
    points = list(range(n))
    attempts = 0
    while len(triples) < max_triples and attempts < 5000:
        attempts += 1
        t = rng.sample(points, 3)
        ok = True
        for e in triples:
            if len(set(t) & set(e)) > 1:
                ok = False
                break
        if ok:
            triples.append(t)
    return triples

def _try_add(triples, new_t, n=16):
    for e in triples:
        if len(set(new_t) & set(e)) > 1:
            return None
    return triples + [new_t]

def construct():
    start = time.time()
    rng = random.Random(12345)
    n = 16

    best_triples = None
    best_alpha = 16

    # Generate several random systems, keep best
    for _ in range(200):
        if time.time() > DEADLINE - 20:
            break
        sys_ = _random_linear_system(rng, n)
        a = _superset_mark(sys_)
        if a < best_alpha:
            best_alpha = a
            best_triples = sys_

    if best_triples is None:
        best_triples = _random_linear_system(rng, n)
        best_alpha = _superset_mark(best_triples)

    # Local search: improve by adding random valid triples
    cur = list(best_triples)
    cur_a = best_alpha
    points = list(range(n))
    no_improve = 0
    while time.time() < DEADLINE - 5:
        # try to add a random triple
        t = rng.sample(points, 3)
        new = _try_add(cur, t, n)
        if new is not None:
            a = _superset_mark(new)
            if a < cur_a:
                cur = new
                cur_a = a
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > 400:
            # try a swap: remove one random triple, add two
            if len(cur) > 1:
                idx = rng.randrange(len(cur))
                cur2 = cur[:idx] + cur[idx+1:]
                added = 0
                for _ in range(2):
                    tt = rng.sample(points, 3)
                    new2 = _try_add(cur2, tt, n)
                    if new2 is not None:
                        cur2 = new2
                        added += 1
                if added == 2:
                    a = _superset_mark(cur2)
                    if a < cur_a:
                        cur = cur2
                        cur_a = a
                        no_improve = 0
            no_improve = 0
        if cur_a <= 6:
            break

    if cur_a < best_alpha:
        best_alpha = cur_a
        best_triples = cur

    return [list(t) for t in best_triples]
