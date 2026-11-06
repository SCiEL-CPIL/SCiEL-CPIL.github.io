"""Strong local-search solver for equal-lcm-triple-free subsets of [1,128]."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _build_conf():
    """conf[x] = list of (a,b), a<b, a,b != x, such that {a,b,x} is a bad triple."""
    n = 128
    L = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            L[i][j] = _lcm(i, j)
            L[j][i] = L[i][j]
    conf = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        lst = []
        for a in range(1, n+1):
            if a == x:
                continue
            for b in range(a+1, n+1):
                if b == x:
                    continue
                if L[a][b] == L[a][x] and L[a][b] == L[b][x]:
                    lst.append((a, b))
        conf[x] = lst
    return conf

def _active_count(x, S, conf):
    c = 0
    for (a, b) in conf[x]:
        if a in S and b in S:
            c += 1
    return c

def _total_active(S, conf):
    t = 0
    for x in range(1, 129):
        if x not in S:
            t += _active_count(x, S, conf)
    return t

def _is_valid(S, conf):
    for x in S:
        for (a, b) in conf[x]:
            if a in S and b in S:
                return False
    return True

def _greedy(order, conf):
    S = set()
    for x in order:
        if all(not (a in S and b in S) for (a, b) in conf[x]):
            S.add(x)
    return S

def _local_search(S, conf, deadline, rng):
    S = set(S)
    while time.time() < deadline:
        # 1) try a single add: pick addable x minimizing new active pairs
        best_x, best_cost = None, None
        for x in range(1, 129):
            if x in S:
                continue
            if _active_count(x, S, conf) == 0:
                # cost = how many new active pairs x creates for others
                cost = 0
                for y in range(1, 129):
                    if y in S or y == x:
                        continue
                    # x becomes a member; count pairs (x, z) in conf[y] with z in S
                    for (a, b) in conf[y]:
                        if (a == x and b in S) or (b == x and a in S):
                            cost += 1
                if best_cost is None or cost < best_cost:
                    best_cost, best_x = cost, x
        if best_x is not None:
            S.add(best_x)
            continue
        # 2) 1-swap that reduces total active count
        cur_total = _total_active(S, conf)
        did_swap = False
        for x in range(1, 129):
            if x in S:
                continue
            for v in list(S):
                S.remove(v)
                if _active_count(x, S, conf) == 0:
                    new_total = _total_active(S, conf)
                    if new_total < cur_total:
                        S.add(x)
                        did_swap = True
                        break
                    else:
                        S.add(v)
                else:
                    S.add(v)
            if did_swap:
                break
        if did_swap:
            continue
        # 3) bounded 2-for-1 move
        tried = 0
        found = False
        while tried < 400 and not found:
            tried += 1
            x1, x2 = rng.sample([x for x in range(1, 129) if x not in S], 2)
            for v in list(S):
                S.remove(v)
                ok = (_active_count(x1, S, conf) == 0 and _active_count(x2, S, conf) == 0
                      and not any((a == x1 and b == x2) or (a == x2 and b == x1)
                                  or (a == x1 and b in S) or (b == x1 and a in S)
                                  or (a == x2 and b in S) or (b == x2 and a in S)
                                  for (a, b) in conf[x1]))
                # simpler: check no bad triple among {x1,x2} with S
                ok2 = True
                for a in S:
                    if _lcm(x1, x2) == _lcm(x1, a) == _lcm(x2, a):
                        ok2 = False
                        break
                if ok and ok2:
                    S.add(x1); S.add(x2)
                    found = True
                    break
                else:
                    S.add(v)
            if found:
                break
        if found:
            continue
        # 4) random perturbation
        k = rng.randint(3, 8)
        for v in rng.sample(list(S), min(k, len(S))):
            S.discard(v)
        # re-greedy fill
        order = list(S)
        rng.shuffle(order)
        for x in range(1, 129):
            if x in S:
                continue
            if all(not (a in S and b in S) for (a, b) in conf[x]):
                S.add(x)
    return S

def construct():
    start = time.time()
    deadline = start + 78.0  # safety margin under 90s kill
    conf = _build_conf()
    rng = random.Random(12345)

    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: (x.bit_count(), x)),
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
    ]
    for _ in range(4):
        o = list(range(1, 129))
        rng.shuffle(o)
        orders.append(o)

    best = set()
    for order in orders:
        if time.time() > deadline:
            break
        seed = _greedy(order, conf)
        res = _local_search(seed, conf, deadline, rng)
        if len(res) > len(best):
            best = set(res)

    # final safety: verify validity, else fall back to a safe greedy
    if not _is_valid(best, conf) or not best:
        best = _greedy(list(range(1, 129)), conf)
    if not best:
        best = {1}
    return list(best)
# EVOLVE-BLOCK-END
