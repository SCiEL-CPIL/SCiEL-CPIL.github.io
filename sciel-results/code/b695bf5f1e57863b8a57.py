"""Maximize subset of [1,180] with no three distinct a,b,c satisfying 1/a = 1/b + 1/c."""
import time
import random

_N = 180

def _build_conf():
    """Pairwise conflict graph: i~j iff they co-occur in a forbidden triple."""
    triples = set()
    for a in range(1, _N + 1):
        a2 = a * a
        for d in range(1, int(a2 ** 0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= _N and c <= _N:
                triples.add(tuple(sorted((a, b, c))))
    conf = [set() for _ in range(_N + 1)]
    for (a, b, c) in triples:
        conf[a].add(b); conf[a].add(c)
        conf[b].add(a); conf[b].add(c)
        conf[c].add(a); conf[c].add(b)
    return conf

def _can_add(x, cur, conf):
    for y in conf[x]:
        if y in cur:
            return False
    return True

def _saturate(cur, conf):
    """Add every element that fits (greedy, order-independent result for independence)."""
    changed = True
    while changed:
        changed = False
        for i in range(1, _N + 1):
            if i not in cur and _can_add(i, cur, conf):
                cur.add(i)
                changed = True
    return cur

def _local_search(cur, conf, deadline):
    """1-flip local search: try removing one element then saturating; keep if better."""
    cur = set(cur)
    _saturate(cur, conf)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elems = list(cur)
        random.shuffle(elems)
        for x in elems:
            if time.time() > deadline:
                break
            cur.remove(x)
            cand = _saturate(cur, conf)
            if len(cand) > len(cur):
                improved = True
            # else keep cur (with x removed) only if better; but we removed x already.
            # If not better, restore x
            else:
                cur.add(x)
                _saturate(cur, conf)
    return cur

def construct():
    start_time = time.time()
    deadline = start_time + 80

    conf = _build_conf()
    deg = [len(conf[i]) for i in range(1, _N + 1)]
    order = sorted(range(1, _N + 1), key=lambda i: (deg[i], i))

    cur = set()
    for i in order:
        if _can_add(i, cur, conf):
            cur.add(i)
    _saturate(cur, conf)

    best = set(cur)
    best_size = len(cur)

    for restart in range(500):
        if time.time() > deadline:
            break
        cand = set(best)
        k = random.randint(1, 5)
        for _ in range(k):
            if cand:
                cand.remove(random.choice(list(cand)))
        cand = _local_search(cand, conf, deadline)
        if len(cand) > best_size:
            best_size = len(cand)
            best = cand

    return list(best)
