import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute base cycle edges (the 40-cycle)
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set()
    for u, v in base_edges:
        if u > v:
            u, v = v, u
        base_edge_set.add((u, v))

    # We add chords from vertex 0 to some subset S of {2, ..., N-1}
    # A chord (0, s) enables:
    #   - cycle of length s+1: [0, 1, ..., s]
    #   - cycle of length N-s+1: [0, N-1, N-2, ..., s]
    #   - for two chords a < b: cycle of length b-a+2: [0, b, b-1, ..., a]
    
    # Target: cover all lengths 3..N
    required = set(range(3, N + 1))
    
    # Length N is always covered by the base cycle
    covered = {N}
    
    # Precompute for each chord endpoint s: which lengths it directly covers
    chord_direct = {}
    for s in range(2, N):
        lengths = set()
        l1 = s + 1
        if 3 <= l1 <= N:
            lengths.add(l1)
        l2 = N - s + 1
        if 3 <= l2 <= N:
            lengths.add(l2)
        chord_direct[s] = lengths
    
    # For a given set S of chord endpoints, compute which lengths are covered
    def compute_covered(S):
        cov = {N}
        S_sorted = sorted(S)
        for s in S_sorted:
            cov.update(chord_direct[s])
        for i in range(len(S_sorted)):
            for j in range(i + 1, len(S_sorted)):
                a = S_sorted[i]
                b = S_sorted[j]
                l = b - a + 2
                if 3 <= l <= N:
                    cov.add(l)
        return cov

    # Beam search for minimal S
    candidates = list(range(2, N))  # 2..39
    
    best_S = None
    best_covered_count = 0
    
    # Try multiple random orderings with beam search
    import random
    rng = random.Random(42)
    
    max_beams = 20
    max_width = 10
    
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Shuffle candidates for diversity
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        
        # Beam search
        beams = [set()]
        best_in_beam = 0
        
        for s in shuffled:
            if time.time() > deadline - 5.0:
                break
            
            new_beams = []
            for beam in beams:
                # Option 1: don't add s
                new_beams.append(beam)
                # Option 2: add s
                new_beam = beam | {s}
                new_beams.append(new_beam)
            
            # Evaluate and keep top max_width
            scored = []
            for nb in new_beams:
                cov = compute_covered(nb)
                count = len(cov & required)
                scored.append((count, len(nb), nb))
            
            scored.sort(key=lambda x: (-x[0], x[1]))
            
            # Keep unique beams
            seen = set()
            unique_scored = []
            for item in scored:
                key = frozenset(item[2])
                if key not in seen:
                    seen.add(key)
                    unique_scored.append(item)
            
            beams = [item[2] for item in unique_scored[:max_width]]
            
            if beams:
                best_count = max(len(compute_covered(b) & required) for b in beams)
                if best_count > best_in_beam:
                    best_in_beam = best_count
            
            # Early termination if we've covered everything
            if best_in_beam == len(required):
                break
        
        # Find the best beam
        for beam in beams:
            cov = compute_covered(beam)
            count = len(cov & required)
            if count > best_covered_count or (count == best_covered_count and len(beam) < len(best_S or [])):
                best_covered_count = count
                best_S = beam
        
        if best_covered_count == len(required):
            break
    
    if best_S is None:
        # Fallback: use all chords
        best_S = set(range(2, N))
    
    S = best_S
    
    # Now we have S. Build the edge list.
    edges = list(base_edges)
    for s in S:
        edges.append((0, s))
    
    # Remove duplicate edges
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges
    
    # Build cycles for each length 3..N
    cycles_by_len = {}
    
    # Hamiltonian cycle
    cycles_by_len[N] = list(range(N))
    
    S_sorted = sorted(S)
    
    # From single chords
    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))
        
        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            # Cycle: 0, N-1, N-2, ..., s
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))
    
    # From pairs of chords
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            l = b - a + 2
            if 3 <= l <= N and l not in cycles_by_len:
                # Cycle: 0, b, b-1, ..., a
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))
    
    # Check if all lengths are covered
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            # Fallback
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L <= 39:
                k = 41 - L
                cycles_by_len[L] = [0] + list(range(39, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))
    
    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]
    
    # Verify all cycles are valid (edges exist)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    valid = True
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid construction
        S_fallback = set(range(2, 22))
        edges_fallback = list(base_edges) + [(0, s) for s in S_fallback]
        unique_edges_fallback = []
        seen_fallback = set()
        for u, v in edges_fallback:
            if u > v:
                u, v = v, u
            if (u, v) not in seen_fallback:
                seen_fallback.add((u, v))
                unique_edges_fallback.append((u, v))
        
        cycles_fallback = []
        for L in range(3, N + 1):
            if L <= 21:
                cycles_fallback.append(list(range(L)))
            elif L <= 39:
                k = 41 - L
                cycles_fallback.append([0] + list(range(39, k - 1, -1)))
            else:
                cycles_fallback.append(list(range(N)))
        
        return {'edges': unique_edges_fallback, 'cycles': cycles_fallback}
    
    return {'edges': edges, 'cycles': final_cycles}
