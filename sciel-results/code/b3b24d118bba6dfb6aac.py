"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Base: 40-cycle 0-1-2-...-39-0
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords from 0 to k for k=2..20, giving cycle lengths k+1 and 41-k
    chords = [(0, k) for k in range(2, 21)]
    
    edges = cycle_edges + chords
    
    # Verify all edges are valid and distinct
    edge_set = set()
    for u, v in edges:
        assert 0 <= u < N and 0 <= v < N
        assert u != v
        key = (min(u, v), max(u, v))
        assert key not in edge_set
        edge_set.add(key)
    
    # Build cycles for each length 3..40
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # The 40-cycle
            cycles.append(list(range(N)))
        else:
            # Find k such that k+1 == length or 41-k == length
            # k+1 == length => k = length-1
            # 41-k == length => k = 41-length
            if length - 1 <= 20:
                k = length - 1
            else:
                k = 41 - length
            # Cycle: 0, 1, 2, ..., k, 0 (length k+1)
            # or 0, k, k+1, ..., 39, 0 (length 41-k)
            if k + 1 == length:
                # Go 0->1->...->k->0
                cyc = list(range(k + 1))
            else:
                # Go 0->k->k+1->...->39->0
                cyc = [0] + list(range(k, N))
            # Verify this is a valid cycle
            assert len(cyc) == length
            assert len(set(cyc)) == length
            # Verify all edges exist
            for i in range(length):
                u, v = cyc[i], cyc[(i + 1) % length]
                key = (min(u, v), max(u, v))
                assert key in edge_set, f"Edge {key} missing for cycle length {length}"
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
