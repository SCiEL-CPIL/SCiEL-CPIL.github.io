def construct():
    start = time.time()
    deadline = start + 55.0
    N = 128

    # ... define helpers ...

    seeds = []
    # high half
    M = 0
    for x in range(65, 129): M |= 1 << x
    seeds.append(M)
    # greedy asc
    M = 0
    for x in range(1, 129):
        if can_add(M, x): M |= 1 << x
    seeds.append(M)
    # greedy desc
    M = 0
    for x in range(128, 0, -1):
        if can_add(M, x): M |= 1 << x
    seeds.append(M)
    # greedy: low then high
    M = 0
    for x in list(range(1,65)) + list(range(65,129)):
        if can_add(M, x): M |= 1 << x
    seeds.append(M)

    best = max(seeds, key=bitcount)  # all valid
    best_sz = bitcount(best)

    # main local search
    best = local_search(best, deadline)
    best_sz = bitcount(best)

    # random restarts
    import random
    rng = random.Random(12345)
    while time.time() < deadline - 1.0:
        # perturb best: remove 1-2 random elements
        bits = setbits(best)
        k = rng.randint(1, 2)
        rem = rng.sample(bits, min(k, len(bits)))
        M = best
        for x in rem: M &= ~(1 << x)
        M = local_search(M, min(deadline - 1.0, time.time() + 5.0))
        if bitcount(M) > best_sz:
            best = M; best_sz = bitcount(M)

    # final safety
    if not is_valid(best):
        # fallback to high half
        M = 0
        for x in range(65, 129): M |= 1 << x
        best = M
    return setbits(best)
