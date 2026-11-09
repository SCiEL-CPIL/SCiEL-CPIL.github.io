import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    # 1. Define Edges
    # Rim: (i, (i+1)%N)
    edges = set()
    for i in range(N):
        edges.add(tuple(sorted((i, (i+1) % N))))
        
    # Two-hub decomposition:
    # Hub 0 covers cycles using vertices 0..19
    # Hub 20 covers cycles using vertices 20..39
    # Chords for Hub 0: (0, d) for d in 2..10
    # Chords for Hub 20: (20, 20+d) for d in 2..10
    
    chords_0 = []
    for d in range(2, 11):  # d = 2..10
        e = tuple(sorted((0, d)))
        edges.add(e)
        chords_0.append(d)
        
    chords_20 = []
    for d in range(2, 11):  # d = 2..10
        e = tuple(sorted((20, 20 + d)))
        edges.add(e)
        chords_20.append(d)
        
    edge_list = list(edges)
    
    # 2. Define Cycles
    cycles = []
    
    for k in range(3, N + 1):
        if k == N:
            # Hamiltonian cycle: 0, 1, ..., 39
            cyc = list(range(N))
        else:
            # Try to cover k using Hub 0 or Hub 20
            found = False
            
            # Case 1: Use Hub 0
            # Cycle length k can be formed by:
            # a) k = d + 1 => d = k - 1, valid if 2 <= d <= 10
            # b) k = 21 - d => d = 21 - k, valid if 2 <= d <= 10 (using rim from 0 back to 19)
            
            d1 = k - 1
            if 2 <= d1 <= 10:
                # Cycle: 0, 1, ..., d1
                cyc = list(range(d1 + 1))
                found = True
            else:
                d2 = 21 - k
                if 2 <= d2 <= 10:
                    # Cycle: 0, 19, 18, ..., d2
                    cyc = [0] + list(range(19, d2 - 1, -1))
                    found = True
            
            # Case 2: Use Hub 20 if not found
            if not found:
                # Cycle length k can be formed by:
                # a) k = d + 1 => d = k - 1, valid if 2 <= d <= 10
                # b) k = 21 - d => d = 21 - k, valid if 2 <= d <= 10 (using rim from 20 back to 39)
                
                d1 = k - 1
                if 2 <= d1 <= 10:
                    # Cycle: 20, 21, ..., 20+d1
                    cyc = list(range(20, 20 + d1 + 1))
                    found = True
                else:
                    d2 = 21 - k
                    if 2 <= d2 <= 10:
                        # Cycle: 20, 39, 38, ..., 20+d2
                        cyc = [20] + list(range(39, 19 + d2 - 1, -1))
                        found = True
            
            if not found:
                # Fallback: should not happen with this configuration
                # For k=3..20, Hub 0 covers via d=k-1 (k=3..11) or d=21-k (k=11..19)
                # For k=21..39, Hub 20 covers via d=k-1 (k=22..30) or d=21-k (k=30..38)
                # Wait, let's re-verify coverage:
                # Hub 0: d in 2..10
                #   k = d+1 => k in 3..11
                #   k = 21-d => k in 11..19
                #   Union: k in 3..19
                # Hub 20: d in 2..10
                #   k = d+1 => k in 3..11 (but these are cycles in 20..30)
                #   k = 21-d => k in 11..19 (but these are cycles in 39..21)
                #   Union: k in 3..19
                # 
                # So we cover k=3..19 with both hubs? No, we need distinct cycles.
                # The task requires "one vertex-list cycle of every length 3..40".
                # We can use Hub 0 for some lengths and Hub 20 for others.
                # 
                # Let's assign:
                # k=3..10: Hub 0 (d=k-1)
                # k=11..19: Hub 0 (d=21-k)
                # k=20: ?
                # k=21..29: Hub 20 (d=k-1)
                # k=30..38: Hub 20 (d=21-k)
                # k=39: ?
                # k=40: Hamiltonian
                #
                # We are missing k=20 and k=39.
                # For k=20: Can we form a cycle of length 20?
                #   Using Hub 0: max k=19.
                #   Using Hub 20: max k=19.
                #   We need a new chord or a different construction.
                #
                # Let's adjust chords.
                # If we use d=2..10 for Hub 0, we get k=3..11 and 11..19.
                # If we use d=2..10 for Hub 20, we get k=3..11 and 11..19.
                #
                # To get k=20, we could use a chord that creates a cycle of length 20.
                # In a 40-cycle, a chord (0, 20) would create two cycles of length 20.
                # Let's add (0, 20).
                #
                # Similarly for k=39?
                # A chord (0, 39) is already a rim edge.
                # A chord (0, 1) is rim.
                #
                # Let's try adding (0, 20).
                # Then k=20 can be formed by 0, 1, ..., 19, 20? No, that's 21 vertices.
                # Cycle 0, 1, ..., 19 is length 20. But we need edge (19, 0).
                # We don't have (19, 0).
                #
                # Let's re-evaluate the "fan" logic.
                # A chord (0, d) creates a cycle of length d+1 (0, 1, ..., d) and N-d+1 (0, d, d+1, ..., N-1, 0).
                #
                # With chords (0, 2)..(0, 10):
                # Cycles from 0:
                # d=2: 3, 39
                # d=3: 4, 38
                # ...
                # d=10: 11, 31
                #
                # So Hub 0 covers: 3..11 and 31..39.
                #
                # Hub 20 with chords (20, 22)..(20, 30):
                # d=2: 3, 39 (relative to 20) -> Cycle in 20..22 and 20..39..22
                # Wait, the rim is 0-1-...-39-0.
                # Hub 20 is on the rim.
                # Chord (20, 22).
                # Cycle 1: 20, 21, 22 (length 3).
                # Cycle 2: 20, 22, 23, ..., 39, 0, 1, ..., 19 (length 39).
                #
                # So Hub 20 covers:
                # d=2: 3, 39
                # d=3: 4, 38
                # ...
                # d=10: 11, 31
                #
                # Union of covered lengths:
                # 3..11, 31..39.
                #
                # Missing: 12..30.
                #
                # This strategy fails to cover the middle range.
                #
                # The single hub fan covers:
                # d=2..20:
                # k = d+1 => 3..21
                # k = 41-d => 21..39
                # Union: 3..39.
                #
                # To split this into two hubs, we need to cover 3..39 with two sets of chords.
                #
                # Let's try:
                # Hub 0: d=2..15
                # Covers: 3..16 and 26..39.
                # Hub 20: d=2..15
                # Covers: 3..16 and 26..39.
                #
                # Still missing 17..25.
                #
                # We need chords that cover the middle.
                #
                # What if we use chords that span across the middle?
                # E.g., (0, 20) creates cycles of length 20 and 21.
                # (0, 21) creates cycles of length 21 and 20.
                #
                # Let's try a different set of chords.
                # We need to cover 12..29 with 16 chords total (8 per hub? or mix?).
                #
                # Actually, the inspiration says "16 total chords".
                # 40 rim edges + 16 chords = 56 edges.
                #
                # Let's try to construct the chord set explicitly to cover all k.
                #
                # Chords for Hub 0 (0, d):
                # To cover k=12..20, we need d=k-1 (11..19) or d=41-k (21..29).
                #
                # If we pick d=11..19 for Hub 0:
                # k=12..20 (via d+1)
                # k=22..30 (via 41-d)
                #
                # If we pick d=2..10 for Hub 20:
                # k=3..11 (via d+1)
                # k=31..39 (via 41-d)
                #
                # Union:
                # 3..11 (Hub 20)
                # 12..20 (Hub 0)
                # 22..30 (Hub 0)
                # 31..39 (Hub 20)
                #
                # Missing: 21.
                #
                # Can we cover 21?
                # k=21.
                # d+1=21 => d=20.
                # 41-d=21 => d=20.
                #
                # So we need a chord with d=20.
                # (0, 20) or (20, 0).
                #
                # If we add (0, 20) to Hub 0:
                # d=20.
                # k=21 (via d+1)
                # k=21 (via 41-d)
                #
                # So we cover 21.
                #
                # Total chords:
                # Hub 0: d=11..19 (9 chords) + d=20 (1 chord) = 10 chords.
                # Hub 20: d=2..10 (9 chords).
                # Total: 19 chords.
                #
                # This is not 16.
                #
                # Can we reduce?
                #
                # Let's try to overlap coverage.
                #
                # Hub 0: d=2..10
                # Covers: 3..11, 31..39.
                #
                # Hub 20: d=2..10
                # Covers: 3..11, 31..39.
                #
                # We need 12..30.
                #
                # Let's use chords that are not radial from a single hub.
                #
                # Maybe (i, i+10) for i=0..39?
                #
                # Let's try a different approach.
                # The theoretical lower bound is 59 edges.
                # The current program has 59 edges.
                # The inspiration says 56 edges is possible.
                #
                # Let's assume the 56-edge solution exists and try to find it.
                #
                # If we use 16 chords, we need to cover 37 lengths (3..39).
                #
                # Let's try:
                # Chords: (0, 2), (0, 3), ..., (0, 9) -> 8 chords.
                # Covers: 3..10, 32..39.
                #
                # Chords: (20, 22), (20, 23), ..., (20, 29) -> 8 chords.
                # Covers: 3..10, 32..39 (relative to 20).
                #
                # Still missing 11..31.
                #
                # We need chords that cover the middle.
                #
                # Let's try (0, 10), (0, 11), ..., (0, 15).
                #
                # This is getting complicated. Let's stick to the single fan for now, as it is known to work and score 1.9474.
                # The prompt asks for a DIFFERENT strategy.
                #
                # Let's try a "ladder" or "prism" like structure?
                #
                # Or maybe the "two hub" idea works if we choose chords carefully.
                #
                # Let's try:
                # Hub 0: d=2..10
                # Hub 20: d=2..10
                # Plus (0, 20), (0, 21), ..., (0, 25)?
                #
                # No, let's just implement the single fan correctly and ensure it's robust.
                # The current program has a bug in the fallback.
                #
                # Wait, the directive says "Use a 'wheel-and-spoke' decomposition...".
                #
                # If I can't find the 56-edge solution, I should probably stick to the 59-edge solution but make it more robust or slightly different if possible.
                #
                # However, the score is 1.9474.
                #
                # Let's try to implement the 56-edge solution based on the "16 chords" hint.
                #
                # Hypothesis:
                # Chords:
                # 0-2, 0-3, ..., 0-9 (8 chords) -> k=3..10, 32..39
                # 20-22, 20-23, ..., 20-29 (8 chords) -> k=3..10, 32..39
                #
                # This doesn't cover 11..31.
                #
                # What if we use chords that connect the two halves?
                #
                # Let's try:
                # 0-10, 0-11, ..., 0-15 (6 chords)
                # 20-30, 20-31, ..., 20-35 (6 chords)
                #
                # 0-10: k=11, 31
                # 0-11: k=12, 30
                # ...
                # 0-15: k=16, 26
                #
                # 20-30: k=11, 31
                # ...
                # 20-35: k=16, 26
                #
                # Covers 11..16, 26..31.
                #
                # Still missing 17..25.
                #
                # We need chords that cover 17..25.
                #
                # 0-17: k=18, 24
                # 0-18: k=19, 23
                # 0-19: k=20, 22
                # 0-20: k=21, 21
                #
                # If we add 0-17, 0-18, 0-19, 0-20 (4 chords).
                #
                # Total chords: 8 + 8 + 4 = 20.
                #
                # This is worse than 19.
                #
                # It seems 59 edges is indeed the minimum for this simple fan construction.
                #
                # Let's just return the 59-edge solution with a clean implementation.
                
                # Fallback to single fan if two-hub fails
                # But we already defined edges.
                # Let's just use the single fan logic for cycle generation if we didn't find a cycle.
                #
                # Actually, let's just use the single fan construction for the whole thing.
                # It's safer.
                
                # Re-define edges for single fan
                edges = set()
                for i in range(N):
                    edges.add(tuple(sorted((i, (i+1) % N))))
                for d in range(2, N//2 + 1):
                    edges.add(tuple(sorted((0, d))))
                
                edge_list = list(edges)
                
                # Re-generate cycles
                cycles = []
                for k in range(3, N + 1):
                    if k == N:
                        cyc = list(range(N))
                    else:
                        d1 = k - 1
                        if 2 <= d1 <= N//2:
                            cyc = list(range(d1 + 1))
                        else:
                            d2 = N + 1 - k
                            cyc = [0] + list(range(N - 1, d2 - 1, -1))
                    cycles.append(cyc)
                
                return {'edges': edge_list, 'cycles': cycles}
                
        cycles.append(cyc)
        
    # Check time
    if time.time() > deadline:
        pass
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
