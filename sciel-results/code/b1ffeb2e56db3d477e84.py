def construct():
    start = time.time()
    deadline = start + 50
    best = None
    best_count = 10**9
    def consider(c):
        nonlocal best, best_count
        if c is not None and _validate(c):
            cnt = len(set(c))
            if cnt < best_count:
                best, best_count = c, cnt
    # fallback (guaranteed-ish valid, few colors)
    for k in (6, 8, 12):
        if time.time() > deadline - 3:
            break
        fb = _backtrack(k, seed=1, time_limit=min(2.0, max(0.3, deadline - time.time() - 2)))
        consider(fb)
        if best_count <= k:
            break
    # small k search
    for k, slice_s in ((3, 22), (4, 12), (5, 6), (6, 2)):
        t0 = time.time()
        t_end = t0 + slice_s
        if time.time() > deadline - 2:
            break
        # greedy
        s = 0
        while time.time() < t_end - 2 and s < 200:
            g = _greedy(k, seed=s)
            consider(g)
            s += 1
        # local search
        s = 0
        while time.time() < t_end - 0.5 and s < 30:
            rem = t_end - 0.5 - time.time()
            if rem < 0.3:
                break
            ls = _local_search(k, seed=s, time_budget=min(3.0, rem))
            consider(ls)
            s += 1
    if best is None:
        best = list(range(N))
    return best
