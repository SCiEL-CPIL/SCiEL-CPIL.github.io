import time

def construct():
    _start = time.time()
    N = 40
    DEADLINE = 85.0  # safety margin before 90s kill

    # Build edge set: 40 rim edges + 18 chords (0,2)..(0,19) = 58 edges
    edges_set = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        edges_set.add((min(a, b), max(a, b)))
    for d in range(2, 20):  # chords (0,2) through (0,19)
        edges_set.add((0, d))

    # Build cycles for lengths 3..40
    cycles = {}

    # Length 40: the rim
    cycles[40] = list(range(N))

    # Lengths 3..20: forward cycles 0,1,2,...,d (length d+1) using chord (0,d)
    for d in range(2, 20):  # d=2 gives length 3, d=19 gives length 20
        fwd = list(range(0, d + 1))
        cycles[len(fwd)] = fwd

    # Length 21: 0,1,2,...,19,20,0 (uses chord (0,19) and rim edges)
    cycles[21] = list(range(0, 21))

    # Lengths 22..39: backward cycles 0,d,d+1,...,39,0 (length 41-d) using chord (0,d)
    # d=19 gives length 22, d=2 gives length 39
    for d in range(19, 1, -1):  # d=19 down to d=2
        bwd = [0] + list(range(d, N))
        if len(bwd) not in cycles:
            cycles[len(bwd)] = bwd

    # Verify all cycles
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
            a, b = cyc[i], cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if valid:
        if time.time() - _start < DEADLINE:
            return {'edges': [list(e) for e in edges_set], 'cycles': [cycles[k] for k in range(3, N + 1)]}

    # Fallback: fan graph (78 edges, still valid)
    fan_edges = set()
    for i in range(1, N):
        fan_edges.add((0, i))
    for i in range(1, N - 1):
        fan_edges.add((i, i + 1))
    cycles_fb = [list(range(k)) for k in range(3, N + 1)]
    return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}
