import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe under 90s kill

    # Base cycle edges: 0-1-2-...-39-0 (40 edges)
    edges = []
    edge_set = set()

    for i in range(N):
        u = i
        v = (i + 1) % N
        e = (min(u, v), max(u, v))
        edges.append(e)
        edge_set.add(e)

    # Add 19 chords from vertex 0 to vertices 2..20
    # Chord (0,d) certifies cycle length d+1 (forward) and 41-d (backward)
    # Together these cover all lengths 3..39
    # We need to cover k in 3..39.
    # For k, need chord d such that k = d+1 or k = 41-d.
    # d = k-1 or d = 41-k.
    # Pairs (d, 41-d) for d in 2..20 cover all k.
    # d=2 covers k=3, 39
    # d=3 covers k=4, 38
    # ...
    # d=19 covers k=20, 22
    # d=20 covers k=21, 21
    # All k from 3 to 39 are covered.
    
    # Check time before adding chords (though it's fast)
    if time.time() > deadline:
        # Fallback: return base cycle (invalid for pancyclic but prevents crash)
        # Actually, if we haven't added chords, we can't satisfy pancyclic.
        # But this shouldn't happen.
        pass

    for d in range(2, 21):
        e = (0, d)
        edges.append(e)
        edge_set.add(e)
        
        if time.time() > deadline:
            break

    # Generate cycles for lengths 3..40
    cycles = []
    for k in range(3, N + 1):
        if time.time() > deadline:
            # If we run out of time, we can't guarantee valid cycles for remaining k.
            # But the loop is O(N^2) at worst, which is tiny.
            # So this guard is mostly for safety.
            pass

        if k == N:
            cycle = list(range(N))
        else:
            # Try forward: 0, 1, 2, ..., k-1 (needs edge (0, k-1))
            d1 = k - 1
            if 2 <= d1 <= 20:
                cycle = list(range(d1 + 1))
            else:
                # Try backward: 0, N-1, N-2, ..., N-k+1 (needs edge (0, N-k+1))
                # The chord is (0, N-k+1). Let d2 = N-k+1.
                # We need d2 in [2, 20].
                # k in [3, 39].
                # If k-1 > 20, then k > 21.
                # d2 = 40 - k + 1 = 41 - k.
                # If k=22, d2=19.
                # If k=39, d2=2.
                # So d2 is always in [2, 19] for k in [22, 39].
                # So this branch is always valid for k > 21.
                d2 = N - (k - 1)
                cycle = [0] + list(range(N - 1, d2 - 1, -1))
        cycles.append(cycle)

    return {'edges': edges, 'cycles': cycles}
