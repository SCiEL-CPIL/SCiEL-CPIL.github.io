def _search(n, time_budget, rng):
    # Initialize random coloring
    m = n*(n-1)//2
    colors = [rng.randint(0,4) for _ in range(m)]
    # Build neighbor bitsets
    neigh = [ [0]*n for _ in range(5) ]
    def build():
        for c in range(5):
            for u in range(n):
                neigh[c][u] = 0
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                idx += 1
                neigh[c][i] |= (1 << j)
                neigh[c][j] |= (1 << i)
    build()
    # Energy
    def energy():
        e = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = neigh[c][u] & neigh[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        e += k*(k-1)//2
        return e
    # Actually energy should be total C4s = (1/2) sum C(k,2). But for optimization, sum C(k,2) is fine (monotonic).
    # Let me use sum C(k,2) as energy (proportional to C4 count).
    def energy_full():
        e = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = neigh[c][u] & neigh[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        e += k*(k-1)//2
        return e
    E = energy_full()
    if E == 0:
        return colors, 0
    # Local search
    start = time.time()
    best = E
    best_colors = colors[:]
    T = 1.0  # temperature
    T_end = 0.01
    # Precompute edge index mapping
    # For flipping, we need to evaluate delta for each of 4 alternative colors.
    def delta_for_flip(u, v, c_from, c_to):
        # broken: C(|N_cfrom(u) & N_cfrom(v)|, 2)
        S = neigh[c_from][u] & neigh[c_from][v]
        ks = S.bit_count()
        broken = ks*(ks-1)//2 if ks>=2 else 0
        # created: C(|N_cto(u) & N_cto(v)|, 2)
        Tset = neigh[c_to][u] & neigh[c_to][v]
        kt = Tset.bit_count()
        created = kt*(kt-1)//2 if kt>=2 else 0
        return created - broken
    # Main loop
    import math
    steps = 0
    while time.time() - start < time_budget:
        # pick random edge
        idx = rng.randint(0, m-1)
        # find (u,v) for idx
        # precompute
        u, v = edge_uv[idx]
        c_from = colors[idx]
        # evaluate all 4 alternatives
        best_delta = 0
        best_c = None
        for c_to in range(5):
            if c_to == c_from: continue
            d = delta_for_flip(u, v, c_from, c_to)
            if best_c is None or d < best_delta:
                best_delta = d
                best_c = c_to
        # SA accept
        if best_delta <= 0 or (rng.random() < math.exp(-best_delta / T)):
            # apply flip
            colors[idx] = best_c
            neigh[c_from][u] &= ~(1 << v)
            neigh[c_from][v] &= ~(1 << u)
            neigh[best_c][u] |= (1 << v)
            neigh[best_c][v] |= (1 << u)
            E += best_delta
            if E < best:
                best = E
                best_colors = colors[:]
                if E == 0:
                    return best_colors, 0
        # cool
        T *= 0.9999
        if T < T_end:
            T = T_end
        steps += 1
    return best_colors, best
