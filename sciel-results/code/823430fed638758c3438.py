import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    base_edges = [(i, (i + 1) % N) for i in range(N)]
    required = set(range(3, N + 1))
    candidates = list(range(2, N - 1))  # 2..38

    # Precompute single chord coverage
    chord_singles = {}
    for s in range(2, N - 1):
        lengths = set()
        l1 = s + 1
        if 3 <= l1 <= N:
            lengths.add(l1)
        l2 = N - s + 1
        if 3 <= l2 <= N:
            lengths.add(l2)
        chord_singles[s] = lengths

    # Precompute pair coverage
    chord_pairs = {}
    for a in range(2, N - 1):
        chord_pairs[a] = {}
        for b in range(a + 1, N - 1):
            l = b - a + 2
            if 3 <= l <= N:
                chord_pairs[a][b] = l
                chord_pairs[b][a] = l

    # Precompute triple coverage: for chords a < b < c, cycle 0-c-b-a-0 has length c-a+2
    # Actually for three chords at positions a, b, c (a<b<c), we can form:
    # 0-a-...-b-...-c-0: length c+1 (uses chords a,b,c but really just needs c)
    # 0-a-(path to b)-(path to c)-0: various
    # Let's think: with chords at a, b, c, we can form cycle 0-a-1-2-...-b-...-c-0
    # That's just chord c. 
    # Better: 0-a-(reverse path to b)-(reverse path to c)-0? No.
    # With chords a, b, c: cycle 0-a-...-b-0 is length b+1 (chord b alone)
    # Cycle 0-a-...-c-0 is length c+1 (chord c alone)
    # Cycle a-...-b-...-c-a? That doesn't go through 0.
    # Actually, for cycles NOT through 0:
    # a-...-b-...-c-...-a: length N - (c-a) ... no
    # a-...-b-...-c-a: this goes a->b (forward, length b-a), b->c (forward, length c-b), c->a (backward, length N-c+a)
    # Total: (b-a) + (c-b) + (N-c+a) = N. That's the full cycle.
    # a->b forward (b-a edges), b->c backward (c-b edges), c->a forward (N-c+a edges)? No that's N.
    # Let me think differently. In the cycle graph with chords from 0:
    # A cycle using chords at positions a, b, c (a<b<c) not through 0:
    # a -> a+1 -> ... -> b (forward, b-a edges, uses no chord)
    # b -> b-1 -> ... -> a (backward, b-a edges, uses no chord) - that's just a 2-chord cycle
    # For 3 chords: a -> ... -> b -> ... -> c -> ... -> a
    # a to b forward: b-a edges
    # b to c forward: c-b edges  
    # c to a backward: N-c+a edges
    # Total: (b-a) + (c-b) + (N-c+a) = N. Full cycle.
    # 
    # Alternative: a -> ... -> c (forward, c-a edges), c -> ... -> b (backward, c-b edges), b -> ... -> a (backward, b-a edges)
    # Total: (c-a) + (c-b) + (b-a) = 2c - 2a. 
    # This uses chords at a and c (to jump) and b is just a vertex on the path.
    # Wait, we need the chords to actually be used. Let me reconsider.
    #
    # In the fan graph (cycle + chords from 0), any simple cycle either:
    # 1. Goes through 0: uses k chords from 0, length = sum of arc lengths
    # 2. Doesn't go through 0: uses only cycle edges, so it's a subpath of the cycle
    #
    # For case 1 with chords at positions s1 < s2 < ... < sk:
    # Cycle: 0 -> s1 -> s2 -> ... -> sk -> 0 (all forward arcs)
    # Length: s1 + (s2-s1) + ... + (N-sk) = N. Full cycle.
    #
    # Cycle: 0 -> s1 -> 0 -> s2 -> ... no, that repeats 0.
    #
    # Actually with chords from 0, a cycle through 0 using chords at s1, s2:
    # 0 -> s1 (chord) -> s1+1 -> ... -> s2 (forward arc) -> 0 (chord)
    # Length: 1 + (s2-s1) + 1 = s2 - s1 + 2
    #
    # With chords at s1, s2, s3 (s1 < s2 < s3):
    # 0 -> s1 (chord) -> s1+1 -> ... -> s2 (forward) -> s2-1 -> ... -> s3? No.
    # 0 -> s1 (chord) -> ... -> s3 (forward) -> 0 (chord): length s3+1 (just chord s3)
    # 0 -> s3 (chord) -> ... -> s1 (backward) -> 0 (chord): length N-s1+1 (just chord s1)
    # 0 -> s1 (chord) -> ... -> s2 (forward) -> ... -> s3 (forward) -> 0 (chord): 
    #   Length: 1 + (s2-s1) + (s3-s2) + 1 = s3 - s1 + 2
    # This uses chords s1 and s3, with s2 on the path.
    #
    # So triple chords don't give new lengths beyond what pairs give!
    # The length depends only on the min and max chord used.

    # So we're limited to: singles give {s+1, N-s+1}, pairs give {b-a+2}
    # We need to cover all of 3..40 with minimum chords.

    # Known good construction: let's try a systematic approach
    # Chord s covers: s+1 and N-s+1 (singles), and for each other chord t: |t-s|+2 (pairs)
    
    # Let's use a smarter greedy with bitmask
    # Map lengths to bits
    length_to_bit = {L: 1 << (L - 3) for L in range(3, N + 1)}
    required_mask = sum(length_to_bit[L] for L in range(3, N + 1))
    
    # Precompute coverage masks
    single_mask = {}
    for s in range(2, N - 1):
        m = 0
        for L in chord_singles[s]:
            m |= length_to_bit[L]
        single_mask[s] = m
    
    # For a set S, coverage = OR of single_mask[s] for s in S, plus pair contributions
    # Pair contribution for (a,b) is length b-a+2, which is a single bit
    
    # Full coverage mask for a set
    def coverage_mask(S):
        m = 0
        S_list = sorted(S)
        for s in S_list:
            m |= single_mask[s]
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                l = b - a + 2
                if 3 <= l <= N:
                    m |= length_to_bit[l]
        return m

    # Incremental: what does adding s to S cover?
    def incremental_mask(S, s):
        m = single_mask[s]
        for t in S:
            l = abs(t - s) + 2
            if 3 <= l <= N:
                m |= length_to_bit[l]
        return m

    # Start with a good seed
    # Try chords that cover many lengths
    # Chord 2: covers 3, 39. Pairs with t: t+2
    # Chord 20: covers 21, 21. Pairs with t: |t-20|+2
    
    best_S = None
    best_size = float('inf')
    
    import random
    rng = random.Random(42)
    
    # Greedy with multiple restarts
    for trial in range(200):
        if time.time() > deadline - 15.0:
            break
        
        S = set()
        covered = 0
        
        # Randomize candidate order
        cands = candidates[:]
        rng.shuffle(cands)
        
        while not (covered & required_mask) == required_mask:
            best_gain = -1
            best_s = None
            for s in cands:
                if s in S:
                    continue
                new_cov = incremental_mask(S, s)
                gain = bin((new_cov & required_mask) & ~covered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s is None or best_gain <= 0:
                break
            S.add(best_s)
            covered |= incremental_mask(S - {best_s}, best_s)
        
        if (covered & required_mask) == required_mask:
            if len(S) < best_size:
                best_size = len(S)
                best_S = S.copy()
                if best_size <= 10:
                    break
    
    if best_S is None:
        best_S = set(range(2, 39, 3))
    
    # Local search
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            S_list = sorted(best_S)
            
            # Try removals
            for i in range(len(S_list)):
                if time.time() > deadline - 10.0:
                    break
                trial_S = best_S - {S_list[i]}
                if (coverage_mask(trial_S) & required_mask) == required_mask:
                    best_S = trial_S
                    improved = True
                    break
            
            # Try swaps
            if not improved and len(S_list) >= 2:
                for i in range(len(S_list)):
                    if time.time() > deadline - 10.0:
                        break
                    for s in candidates:
                        if s in best_S:
                            continue
                        trial_S = (best_S - {S_list[i]}) | {s}
                        if len(trial_S) == len(best_S) and (coverage_mask(trial_S) & required_mask) == required_mask:
                            best_S = trial_S
                            improved = True
                            break
                    if improved:
                        break
    
    S = best_S
    S_sorted = sorted(S)
    
    # Build edges
    edges = list(base_edges)
    for s in S:
        edges.append((0, s))
    
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges
    
    # Build cycles
    cycles_by_len = {}
    cycles_by_len[N] = list(range(N))
    
    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))
        
        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))
    
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            l = b - a + 2
            if 3 <= l <= N and l not in cycles_by_len:
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))
    
    # Check coverage and add fallbacks
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L <= 39:
                k = 41 - L
                cycles_by_len[L] = [0] + list(range(39, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))
    
    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]
    
    # Verify
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                return {
                    'edges': base_edges + [(0, k) for k in range(2, 21)],
                    'cycles': [
                        list(range(L)) if L <= 21 else ([0] + list(range(39, 41 - L - 1, -1)) if L <= 39 else list(range(N)))
                        for L in range(3, N + 1)
                    ]
                }
    
    return {'edges': edges, 'cycles': final_cycles}
