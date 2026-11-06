N = 40

def construct():
    # 40-cycle: 0-1-2-...-39-0
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords from 0 to 2, 3, ..., 20
    chord_edges = [(0, k) for k in range(2, 21)]
    
    edges = cycle_edges + chord_edges
    
    # Verify no duplicate edges
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        assert key not in edge_set, f"Duplicate edge {key}"
        edge_set.add(key)
    
    # Build cycles of length k for k = 3, 4, ..., 40
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            # Short cycle: 0, 1, 2, ..., k-1
            cyc = list(range(k))
        else:
            # Long cycle using chord (0, 41-k)
            # 41-k ranges from 20 (k=21) down to 2 (k=39)
            chord_target = 41 - k
            # Cycle: 0, chord_target, chord_target+1, ..., 39
            if chord_target == 0:
                cyc = list(range(N))
            else:
                cyc = [0] + list(range(chord_target, N))
        cycles.append(cyc)
    
    # Verify cycles
    for i, cyc in enumerate(cycles):
        k = i + 3
        assert len(cyc) == k, f"Cycle {k} has length {len(cyc)}"
        assert len(set(cyc)) == k, f"Cycle {k} has duplicate vertices"
        for j in range(k):
            u = cyc[j]
            v = cyc[(j + 1) % k]
            key = (min(u, v), max(u, v))
            assert key in edge_set, f"Cycle {k}: edge ({u},{v}) missing"
    
    return {'edges': edges, 'cycles': cycles}
