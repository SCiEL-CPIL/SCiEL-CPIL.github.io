import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Base cycle (rim)
    rim_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # Chords to cover all cycle lengths 3..39
    # Set K = {2, 3, ..., 17, 19, 20}
    # This set has 18 elements.
    # Single chords cover:
    # k in 2..17 -> 3..18 and 39..24
    # k=19 -> 20, 22
    # k=20 -> 21
    # Multi-chords (diff d) cover d+2 and 40-d
    # We need to ensure 19 and 23 are covered.
    # 19 requires d=17 or d=21.
    # 23 requires d=21 or d=17.
    # Max diff in K is 18. Min is 1.
    # Pairs with diff 17: (3, 20), (2, 19).
    # So 19 and 23 are covered.
    
    chords = [(0, k) for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]]
    
    edges = rim_edges + chords
    
    cycles = []
    
    # Map length to cycle
    # L=40: Rim
    # L in 3..39:
    # Check single chords
    # Check multi chords
    
    # Precompute single chord cycles
    single_cycles = {}
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]:
        # Cycle 0..k
        L1 = k + 1
        if L1 not in single_cycles:
            single_cycles[L1] = list(range(k + 1))
        # Cycle 0, k, k-1, ..., 1? No, 0, 39, ..., k
        L2 = N - k + 1
        if L2 not in single_cycles:
            single_cycles[L2] = [0] + list(range(k, N, -1)) # 0, k, k-1... wait
            # Rim path k to 0 via 39: k, k-1... no, k, k+1...39, 0
            # Vertices: 0, k, k+1, ..., 39
            # Length: 1 + (39-k+1) = 40-k+1 = 41-k
            # My L2 formula was N-k+1 = 41-k. Correct.
            # Vertices: 0, then k to 39.
            # list: [0] + list(range(k, N))
            single_cycles[L2] = [0] + list(range(k, N))
            
    # Precompute multi chord cycles
    # We only need to fill gaps.
    # Gaps: 19, 23 (and potentially others if single missed)
    # But we verified singles cover 3..18, 20..22, 24..39.
    # So only 19, 23 are missing.
    
    # Find pair (a,b) with diff 17 for L=19 and L=23.
    # a=3, b=20. Diff 17.
    # Cycle 1: 0, 3, 4, ..., 20. Length 20-3+2 = 19.
    # Cycle 2: 0, 20, 21, ..., 39, 1, 2, 3? No.
    # Cycle 2 uses other rim segment.
    # Vertices: 0, 20, 21, ..., 39, 1, 2, 3.
    # Length: 1 + (39-20+1) + (3-1+1) = 1 + 20 + 3 = 24?
    # Wait.
    # Rim segment 20 to 3 via 0: 20..39, 0, 1..3.
    # Vertices: 20..39 (20), 0 (1), 1,2,3 (3). Total 24.
    # But we need 23.
    # Formula: N - (b-a) = 40 - 17 = 23.
    # Why did I count 24?
    # Vertices in cycle: 0, 20, 21, ..., 39, 1, 2, 3.
    # Is 0 connected to 3? Yes, chord.
    # Is 3 connected to 2? Yes, rim.
    # Is 2 connected to 1? Yes.
    # Is 1 connected to 0? Yes, rim.
    # Is 0 connected to 20? Yes, chord.
    # Path: 0-20-21-...-39-1-2-3-0?
    # No, 39 is connected to 0.
    # So 39-0 is an edge.
    # So path 20..39-0-1-2-3-0.
    # Vertices: 20..39 (20), 0 (1), 1,2,3 (3).
    # Total 24 vertices.
    # Length 24.
    # But formula said 23.
    # Let's re-verify length of Cycle 2.
    # Cycle 2: 0-b-...-a-0.
    # Vertices: 0, b, b+1, ..., 39, 1, ..., a.
    # Count: 1 + (39-b+1) + (a-1+1) = 1 + 40-b + a = 41 - (b-a).
    # For b=20, a=3.
    # 41 - 17 = 24.
    # So Cycle 2 has length 24.
    # But 24 is already covered by single chord k=17.
    # We need 23.
    # So this pair (3,20) gives 19 and 24.
    # It does NOT give 23.
    
    # We need a pair that gives 23.
    # L=23 requires d=17 (from 41-d=23 -> d=18? No 41-18=23).
    # Or d=21 (from d+2=23).
    # If d=18, L1=20, L2=23.
    # Do we have diff 18?
    # (2, 20) diff 18.
    # Cycle 1: 0, 2, ..., 20. Length 20. (Covered by single 19).
    # Cycle 2: 0, 20, ..., 39, 1, 2.
    # Length: 41 - 18 = 23.
    # Vertices: 0, 20..39 (20), 1, 2 (2). Total 23.
    # Yes!
    
    # So pair (2, 20) gives 20 and 23.
    # Pair (3, 20) gives 19 and 24.
    
    # So we use (3,20) for 19.
    # And (2,20) for 23.
    
    if 19 not in single_cycles:
        # Use a=3, b=20
        cyc = [0] + list(range(3, 21)) # 0, 3, ..., 20
        single_cycles[19] = cyc
        
    if 23 not in single_cycles:
        # Use a=2, b=20
        # Cycle 2: 0, 20, 21, ..., 39, 1, 2
        cyc = [0] + list(range(20, N)) + [1, 2]
        # Wait, 0 is already in list? No, list starts with 0.
        # 0, 20, 21, ..., 39, 1, 2.
        # Edges: (0,20), (20,21)...(39,1)? No (39,0).
        # (39,0) is rim edge.
        # (0,1) is rim edge.
        # (1,2) is rim edge.
        # (2,0) is chord.
        # So 0-20-...-39-0-1-2-0.
        # This is a valid cycle.
        single_cycles[23] = cyc

    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        else:
            cyc = single_cycles.get(L, [])
            if not cyc:
                # Fallback, should not happen
                cyc = list(range(L))
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
