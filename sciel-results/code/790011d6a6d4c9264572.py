import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0
    N = 40

    # Rim edges
    edges_set = set()
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))

    # Chords from vertex 0 to cover cycle lengths 3..39
    # Chord (0,d) gives cycles of length d+1 and N-d+1
    # d=2..20 covers lengths 3..21 and 39..21
    for d in range(2, 21):
        edges_set.add((0, d))

    # Build explicit cycles
    cycles = {}
    cycles[40] = list(range(N))  # The rim

    for d in range(2, 21):
        # Forward cycle: 0,1,2,...,d (length d+1)
        fwd = list(range(0, d + 1))
        # Backward cycle: 0,d,d+1,...,39 (length N-d+1)
        bwd = [0] + list(range(d, N))

        if 3 <= len(fwd) <= 40 and len(fwd) not in cycles:
            cycles[len(fwd)] = fwd
        if 3 <= len(bwd) <= 40 and len(bwd) not in cycles:
            cycles[len(bwd)] = bwd

    # Ensure length 21 is covered (should be from d=20: fwd=[0..20] len 21, bwd=[0,20..39] len 21)
    if 21 not in cycles:
        cycles[21] = [0] + list(range(1, 21))

    # Verify all cycles are valid
    valid = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k or len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if not valid or time.time() > DEADLINE:
        # Fallback: fan graph (star + path)
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}

    out = [cycles[k] for k in range(3, N + 1)]
    return {'edges': [list(e) for e in edges_set], 'cycles': out}
