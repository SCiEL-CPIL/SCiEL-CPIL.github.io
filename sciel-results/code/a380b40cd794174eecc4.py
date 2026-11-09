import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the 40-cycle rim
    edges_set = set()
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))

    # Chord set: distances 2..18 (17 chords)
    # This covers cycle lengths:
    #   d+1: 3,4,5,...,19
    #   N-d+1: 39,38,37,...,23
    # Missing: 20, 21, 22
    # We add chord (0,20) for length 21 (both paths give 21)
    # For lengths 20 and 22, we use composite cycles:
    #   Length 20: 0,2,3,...,20 (uses chords (0,2) and (0,20))
    #   Length 22: 0,2,3,...,22 (uses chords (0,2) and (0,22) -> need chord (0,22))
    # Actually, let's think more carefully.
    
    # With chords at distances 2..18:
    # Forward cycles: d+1 for d=2..18 -> lengths 3..19
    # Backward cycles: N-d+1 for d=2..18 -> lengths 39..23
    # So we have lengths 3..19 and 23..39, plus 40 (rim)
    # Missing: 20, 21, 22
    
    # Add chord (0,20): gives cycle of length 21 (both paths)
    # Add chord (0,22): gives cycle of length 23 (already have) and 19 (already have)
    #   But we can make composite cycle for 22: 0,2,3,...,22 using chords (0,2) and (0,22)
    # Add chord (0,18): already have, gives 19 and 23
    # For length 20: 0,2,3,...,20 using chords (0,2) and (0,20)
    # For length 21: 0,1,2,...,20 using chord (0,20) - simple cycle
    # For length 22: 0,2,3,...,22 using chords (0,2) and (0,22)
    
    # So we need chords at distances: 2,3,4,...,18, 20, 22
    # That's 17 + 2 = 19 chords total
    # Total edges: 40 + 19 = 59, objective = 19
    
    # Wait, can we do better? Let's try to minimize chords.
    # We need to cover lengths 3..39 (37 lengths) plus 40.
    # Each chord (0,d) gives at most 2 cycle lengths: d+1 and N-d+1.
    # If d+1 == N-d+1, i.e., d = N/2 = 20, it gives only 1 length (21).
    # So we need at least ceil(37/2) = 19 chords if all are non-self-complementary.
    # But some chords give the same length, so we need to be careful.
    
    # Actually, the current program has 18 chords (distances 2..18 plus 21) and uses a composite cycle for 21.
    # Let me verify: distances 2..18 is 17 chords, plus distance 21 is 18 chords.
    # With these chords:
    #   Forward: d+1 for d=2..18 -> 3..19 (17 lengths)
    #   Backward: N-d+1 for d=2..18 -> 39..23 (17 lengths)
    #   Chord 21: forward 22, backward 20
    #   Composite: 21 using chords 2 and 21
    # So we have: 3..19, 20, 21, 22, 23..39, 40 = all 38 lengths (3..40)
    # Total edges: 40 + 18 = 58, objective = 18
    
    # Can we do 17 chords? We'd need to cover 37 lengths (3..39) with 17 chords.
    # 17 chords * 2 = 34 max, so we need at least 3 composite cycles or some chords giving 3+ lengths.
    # This seems difficult. Let's stick with 18 chords for now.
    
    chord_distances = list(range(2, 19)) + [21]
    
    chords = []
    for d in chord_distances:
        u = 0
        v = d % N
        if u == v:
            continue
        e = norm_edge(u, v)
        if e not in edges_set:
            edges_set.add(e)
            chords.append((u, v, d))

    # Now extract explicit cycles for each length 3-40
    all_cycles = {}
    all_cycles[N] = list(range(N))
    
    # Helper to get cycle from chord d (simple)
    def get_simple_cycle(d, direction):
        if direction == 'forward':
            return list(range(0, d + 1))
        else:
            return [0] + list(range(d, N))

    # Process simple cycles from chords
    for u, v, d in chords:
        len_a = d + 1
        len_b = N - d + 1
        
        if 3 <= len_a <= N:
            if len_a not in all_cycles:
                all_cycles[len_a] = get_simple_cycle(d, 'forward')
        
        if 3 <= len_b <= N:
            if len_b not in all_cycles:
                all_cycles[len_b] = get_simple_cycle(d, 'backward')

    # Handle length 21 specifically with composite cycle
    has_2 = (0, 2) in edges_set
    has_21 = (0, 21) in edges_set
    
    if 21 not in all_cycles:
        if has_2 and has_21:
            cyc_21 = [0] + list(range(2, 22))
            valid = True
            for i in range(len(cyc_21)):
                a = cyc_21[i]
                b = cyc_21[(i + 1) % len(cyc_21)]
                if norm_edge(a, b) not in edges_set:
                    valid = False
                    break
            if valid:
                all_cycles[21] = cyc_21

    # Verify all cycles are found and valid
    valid = True
    for k in range(3, N + 1):
        if all_cycles.get(k) is None:
            valid = False
            break
        cyc = all_cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if norm_edge(a, b) not in edges_set:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        cycles_out = [all_cycles[k] for k in range(3, N + 1)]
        edges_out = [list(e) for e in edges_set]
        return {'edges': edges_out, 'cycles': cycles_out}
    else:
        # Fallback: fan graph
        edges_set = set()
        for i in range(1, N):
            edges_set.add(norm_edge(0, i))
        for i in range(1, N - 1):
            edges_set.add(norm_edge(i, i + 1))
        cycles_fallback = []
        for k in range(3, N + 1):
            cycles_fallback.append(list(range(k)))
        return {'edges': [list(e) for e in edges_set], 'cycles': cycles_fallback}
