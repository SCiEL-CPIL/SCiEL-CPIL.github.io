# EVOLVE-BLOCK-START
import time
import random as _rand

_DEADLINE = 110.0  # wall-clock safety margin (limit is 150s)
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _instance(size, seed, limit=10000):
    from random import Random
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict(vals):
    s = set(vals)
    idx = {v: i for i, v in enumerate(vals)}
    n = len(vals)
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = vals[i]
        for j in range(n):
            c = a + vals[j]
            if c in s:
                k = idx[c]
                if k != i:
                    adj[i].add(k)
                if k != j:
                    adj[j].add(k)
    return adj

def _conflicts(i, adj, present):
    cnt = 0
    for k in adj[i]:
        if present[k]:
            cnt += 1
    return cnt

def _greedy(vals, adj, rng, mode):
    n = len(vals)
    present = [False] * n
    if mode == 0:
        order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    elif mode == 1:
        order = sorted(range(n), key=lambda i: len(adj[i]))
    else:
        order = list(range(n))
        rng.shuffle(order)
    for i in order:
        if not present[i] and _conflicts(i, adj, present) == 0:
            present[i] = True
    return present

def _sa(vals, adj, present, rng, time_budget):
    n = len(vals)
    present = list(present)
    cur_size = sum(present)
    best = list(present)
    best_size = cur_size
    T = 2.0
    T_end = 0.05
    t0 = time.time()
    total_steps = 0
    while time.time() - t0 < time_budget:
        total_steps += 1
        r = rng.random()
        if r < 0.4:
            # single add
            cand = rng.randrange(n)
            if present[cand]:
                continue
            delta = -_conflicts(cand, adj, present)
            if delta >= 0 or rng.random() < _boltzmann(T, delta):
                present[cand] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best = list(present)
        elif r < 0.7:
            # single remove
            cand = rng.randrange(n)
            if not present[cand]:
                continue
            delta = _conflicts(cand, adj, present)
            if delta >= 0 or rng.random() < _boltzmann(T, delta):
                present[cand] = False
                cur_size -= 1
        else:
            # swap: remove a present, add an absent
            if cur_size == 0 or cur_size == n:
                continue
            # pick a present to remove
            pres = [i for i in range(n) if present[i]]
            if not pres:
                continue
            rem = pres[rng.randrange(len(pres))]
            # pick an absent to add
            abs_list = [i for i in range(n) if not present[i]]
            if not abs_list:
                continue
            add = abs_list[rng.randrange(len(abs_list))]
            # delta: removing rem frees conflicts, adding add creates conflicts
            rem_conf = _conflicts(rem, adj, present)
            # after removing rem, recompute conflicts for add
            present[rem] = False
            add_conf = _conflicts(add, adj, present)
            delta = rem_conf - add_conf
            if delta >= 0 or rng.random() < _boltzmann(T, delta):
                present[add] = True
                cur_size += 0  # remove then add => net 0
                if cur_size > best_size:
                    best_size = cur_size
                    best = list(present)
            else:
                present[rem] = True
        # cool
        T *= 0.9995
        if T < T_end:
            T = T_end
    return best, best_size

def _boltzmann(T, delta):
    if delta >= 0:
        return 1.0
    if T <= 0:
        return 0.0
    return min(1.0, 0.1 ** (-delta / max(T, 1e-9)))

def _solve_instance(vals):
    n = len(vals)
    adj = _build_conflict(vals)
    rng = _rand.Random(12345)
    # try several greedy starts, keep the best
    best_present = None
    best_size = -1
    for mode in (0, 1, 2, 2, 2, 2, 3, 3):
        p = _greedy(vals, adj, rng, mode)
        sz = sum(p)
        if sz > best_size:
            best_size = sz
            best_present = p
    # simulated annealing from the best greedy start
    p, sz = _sa(vals, adj, best_present, rng, time_budget=6.0)
    if sz > best_size:
        best_size = sz
        best_present = p
    return [vals[i] for i in range(n) if best_present[i]]

def construct():
    global _START
    _START = time.time()
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if _remaining() < 20.0:
            # not enough time; fall back to odd numbers
            vals = _instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = _instance(size, seed)
        sub = _solve_instance(vals)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
