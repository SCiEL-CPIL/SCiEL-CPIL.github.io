def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 88.0

    suffix = [0.0]*(_N+2)
    for i in range(_N, 0, -1):
        suffix[i] = suffix[i+1] + _RECIP[i]

    best_val = -1.0
    best_set = []

    def consider(S):
        nonlocal best_val, best_set
        if not S: return
        if not _verify(S): return
        v = _val(S)
        if v > best_val:
            best_val = v
            best_set = sorted(S)

    # greedy ascending
    S = set()
    for x in range(1, _N+1):
        if _can_add(S, x):
            S.add(x)
    S = _local_search(S)
    consider(S)

    # random restarts
    rng = random.Random(20240501)
    attempts = 0
    while time.time() < _DEADLINE - 30 and attempts < 12:
        attempts += 1
        S0 = set()
        for x in range(1, _N+1):
            if rng.random() < 0.25 and _can_add(S0, x):
                S0.add(x)
        S = _local_search(S0)
        consider(S)

    # B&B polish
    if time.time() < _DEADLINE - 4 and best_set:
        bs = list(best_set)
        bv, bs = _bb(best_val, bs, suffix)
        if _verify(set(bs)):
            consider(set(bs))

    if not best_set:
        best_set = [1]
    return best_set
