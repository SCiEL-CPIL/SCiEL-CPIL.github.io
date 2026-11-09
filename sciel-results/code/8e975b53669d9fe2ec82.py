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

    # New chord set: distances 2..18 and 21
    # This allows composite cycle for length 21 using chords 2 and 21
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
    cycles = []
    
    # For length 40: the full rim cycle
    all_cycles = {}
    all_cycles[N] = list(range(N))
    
    # Map lengths to cycle definitions
    # We will build them explicitly
    
    # Helper to get cycle from chord d (simple)
    def get_simple_cycle(d, direction):
        if direction == 'forward':
            # 0, 1, ..., d
            return list(range(0, d + 1))
        else:
            # 0, d, d+1, ..., 39
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
    # Vertices: 0, 2, 3, ..., 21
    # Check if we have chords (0,2) and (0,21)
    has_2 = (0, 2) in edges_set or (2, 0) in edges_set
    has_21 = (0, 21) in edges_set or (21, 0) in edges_set
    
    if 21 not in all_cycles:
        if has_2 and has_21:
            # Cycle: 0-2-3-...-21-0
            cyc_21 = [0] + list(range(2, 22))
            # Verify edges
            valid = True
            for i in range(len(cyc_21)):
                a = cyc_21[i]
                b = cyc_21[(i + 1) % len(cyc_21)]
                if norm_edge(a, b) not in edges_set:
                    valid = False
                    break
            if valid:
                all_cycles[21] = cyc_21
            else:
                # Fallback if verification fails (should not happen)
                all_cycles[21] = None
        else:
            all_cycles[21] = None

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
        # Fallback to previous 19-chord solution if this one fails
        edges_set = set()
        for i in range(N):
            edges_set.add(norm_edge(i, (i + 1) % N))
        for d in range(2, 21):
            e = norm_edge(0, d)
            edges_set.add(e)
            
        all_cycles = {}
        all_cycles[N] = list(range(N))
        for d in range(2, 21):
            len_a = d + 1
            len_b = N - d + 1
            if 3 <= len_a <= N:
                all_cycles[len_a] = list(range(0, d + 1))
            if 3 <= len_b <= N:
                all_cycles[len_b] = [0] + list(range(d, N))
                
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
            # Final fallback: fan graph
            edges_set = set()
            for i in range(1, N):
                edges_set.add(norm_edge(0, i))
            for i in range(1, N - 1):
                edges_set.add(norm_edge(i, i + 1))
            cycles_fallback = []
            for k in range(3, N + 1):
                cycles_fallback.append(list(range(k)))
            return {'edges': [list(e) for e in edges_set], 'cycles': cycles_fallback}
