import time
import random

N = 128
DEADLINE = 50.0  # safety margin under 60s

def _full_scan(Sset):
    deg = {v:0 for v in Sset}
    total = 0
    lst = sorted(Sset)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = lst[k]
                if (a + c) in Sset and (b + c) in Sset:
                    total += 1
                    deg[a] += 1
                    deg[b] += 1
                    deg[c] += 1
    return total, deg

def _add(Sset, deg, total, x):
    lst = sorted(Sset)
    c = 0
    for i in range(len(lst)):
        a = lst[i]
        if (x + a) not in Sset:
            continue
        for j in range(i+1, len(lst)):
            b = lst[j]
            if (x + b) in Sset and (a + b) in Sset:
                c += 1
                deg[a] += 1
                deg[b] += 1
    deg[x] = c
    Sset.add(x)
    total += c
    return Sset, total, deg

def _remove(Sset, deg, total, x):
    others = [v for v in Sset if v != x]
    c = 0
    for i in range(len(others)):
        a = others[i]
        if (x + a) not in Sset:
            continue
        for j in range(i+1, len(others)):
            b = others[j]
            if (x + b) in Sset and (a + b) in Sset:
                c += 1
                deg[a] -= 1
                deg[b] -= 1
    total -= c
    Sset.discard(x)
    deg.pop(x, None)
    return Sset, total, deg

def _greedy_delete(S0, tie):
    Sset, total, deg = _full_scan(set(S0))
    guard = 0
    while total > 0 and guard < 1000:
        guard += 1
        if tie == 'random':
            mx = max(deg.values())
            cands = [v for v,d in deg.items() if d == mx]
            v = random.choice(cands)
        elif tie == 'largest':
            v = max(deg.items(), key=lambda kv: (kv[1], kv[0]))[0]
        else:  # smallest
            v = max(deg.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        Sset, total, deg = _remove(Sset, deg, total, v)
    return Sset, total, deg

def _add_back(Sset, deg, total):
    improved = True
    while improved:
        improved = False
        for x in range(1, N+1):
            if x in Sset:
                continue
            lst = sorted(Sset)
            c = 0
            for i in range(len(lst)):
                a = lst[i]
                if (x + a) not in Sset:
                    continue
                for j in range(i+1, len(lst)):
                    b = lst[j]
                    if (x + b) in Sset and (a + b) in Sset:
                        c = 1
                        break
                if c:
                    break
            if c == 0:
                Sset, deg, total = _add(Sset, deg, total, x)
                improved = True
    return Sset, deg, total

def _swap_improve(Sset, deg, total, max_iter=200):
    # try swaps that keep validity and then allow add-back to grow
    best = len(Sset)
    it = 0
    while it < max_iter:
        it += 1
        if time.time() > DEADLINE - 5:
            break
        found = False
        # pick a random x not in S, random y in S
        notin = [x for x in range(1, N+1) if x not in Sset]
        if not notin:
            break
        x = random.choice(notin)
        y = random.choice(list(Sset))
        # remove y
        S2, t2, d2 = _remove(Sset, deg, total, y)
        # try add x
        c = 0
        lst = sorted(S2)
        for i in range(len(lst)):
            a = lst[i]
            if (x + a) not in S2:
                continue
            for j in range(i+1, len(lst)):
                b = lst[j]
                if (x + b) in S2 and (a + b) in S2:
                    c = 1
                    break
            if c:
                break
        if c == 0:
            S2, t2, d2 = _add(S2, d2, t2, x)
            # add-back
            S2, t2, d2 = _add_back(S2, d2, t2)
            if len(S2) > best:
                best = len(S2)
                Sset, deg, total = S2, d2, t2
                found = True
        if not found and it > 50:
            # try a few more then stop if no progress
            pass
    return Sset, deg, total

def construct():
    start = time.time()
    best_set = set(range(N//2+1, N+1))  # sum-free seed size 64
    best_size = len(best_set)
    rng = random.Random(12345)
    ties = ['largest', 'smallest', 'random']
    initial_sets = [
        list(range(1, N+1)),
        list(range(N//2+1, N+1)),
        list(range(1, N//2+1)),
    ]
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        tie = ties[attempts % len(ties)]
        base = initial_sets[attempts % len(initial_sets)]
        # random perturbation for diversity
        S0 = list(base)
        if attempts > 3 and rng.random() < 0.5:
            # random subset of size ~70
            S0 = rng.sample(range(1, N+1), 70)
        Sset, total, deg = _greedy_delete(S0, tie)
        Sset, total, deg = _add_back(Sset, deg, total)
        Sset, total, deg = _swap_improve(Sset, deg, total, max_iter=30)
        if len(Sset) > best_size:
            best_size = len(Sset)
            best_set = set(Sset)
        if time.time() > DEADLINE - 5:
            break
    # final safety: verify best_set valid
    # (trust incremental, but do a quick verify)
    res = sorted(best_set)
    # ensure types are int
    return [int(x) for x in res]
