import time

N = 40
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    edges = []
    for i in range(N):
        u = i
        v = (i + 1) % N
        edges.append((u, v))
    
    # Chords from vertex 0 to cover cycle lengths 3..39
    # We'll use chords to vertex 0 that create cycles with the base cycle
    # Chord (0, k) creates cycles of length k+1 (path 0->1->...->k->0) 
    # and N-k+1 (path 0->39->...->k->0)
    
    # To minimize edges, we want the minimum number of chords such that
    # every length 3..39 is covered by some chord's cycle lengths.
    
    # A chord to vertex k covers lengths: k+1 and N-k+1 = 41-k
    # We need to cover {3, 4, ..., 39}
    
    # Let's try to find a small set of chords
    # Length L is covered if there exists k in chords such that k+1 = L or 41-k = L
    # i.e., k = L-1 or k = 41-L
    
    # So length L is covered if (L-1) is in chords or (41-L) is in chords
    
    # We want to select a set S of chords (vertices 1..39, but 1 and 39 are on the base cycle)
    # Actually, chord (0,1) and (0,39) are already in the base cycle, so they don't add edges.
    # Chords to vertices 2..38 add new edges.
    
    # Let's try to find minimum chords.
    # Lengths 3..39 need to be covered.
    # For each L in 3..39, we need L-1 in S or 41-L in S.
    # Note: L-1 ranges from 2 to 38, and 41-L ranges from 38 to 2.
    # So we're selecting from {2, 3, ..., 38}.
    
    # Let's try a greedy or known construction.
    # If we pick chords to vertices 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20:
    # That's 19 chords, covering lengths 3..21 (via k+1) and 21..39 (via 41-k).
    # Lengths covered: from k=2: 3, 39; k=3: 4, 38; ... k=20: 21, 21.
    # Union: 3..21 from k+1, 21..39 from 41-k. So 3..39. 
    # 19 chords. Total edges = 40 + 19 = 59.
    
    # Can we do better?
    # What if we skip some? Let's see if we can cover with fewer.
    # This is a set cover problem. Let's try to find a minimum.
    
    # Actually, let's just use the 19-chord solution for now, which is known to work.
    # But wait, the current program uses chords to 2..20, which is 19 chords.
    # Let me verify: k=2..20 is 19 values. Yes.
    
    # Let me try to see if we can do with fewer by being clever.
    # Actually, for the purpose of this task, let's stick with a known good solution.
    
    # Let me try chords: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    # That's 19 chords.
    
    # Let me try to find if 18 works.
    # We need to cover all L in 3..39.
    # Let's write a quick check.
    
    def can_cover(chord_set):
        """Check if chord_set covers all cycle lengths 3..39"""
        covered = set()
        for k in chord_set:
            covered.add(k + 1)      # path 0->1->...->k->0
            covered.add(N - k + 1)   # path 0->N-1->...->k->0
        for L in range(3, N + 1):
            if L not in covered:
                return False, L
        return True, None
    
    # Try to find minimum chords using greedy set cover
    # Each chord k covers {k+1, N-k+1}
    # We need to cover {3, 4, ..., 39}
    
    target_lengths = set(range(3, N + 1))
    available_chords = set(range(2, N))  # vertices 2..39, but 39 is same as 1 on cycle
    # Actually chord to 39: k=39, covers 40 and 2. Length 40 is Hamiltonian (already covered), length 2 is not a cycle.
    # So chord to 39 only adds edge but covers nothing new for 3..39.
    # Similarly chord to 1: k=1, covers 2 and 40. Nothing new.
    # So useful chords are k in 2..38.
    
    available_chords = set(range(2, 39))
    
    # Greedy set cover
    remaining = set(target_lengths)
    selected = set()
    
    while remaining:
        best_chord = None
        best_cover = set()
        for k in available_chords:
            if k in selected:
                continue
            cover = set()
            if k + 1 in remaining:
                cover.add(k + 1)
            if N - k + 1 in remaining:
                cover.add(N - k + 1)
            if len(cover) > len(best_cover):
                best_cover = cover
                best_chord = k
            if len(cover) == 2 and len(best_cover) == 2:
                # Tie, pick smaller k for determinism
                if best_chord is None or k < best_chord:
                    best_chord = k
                    best_cover = cover
                break
        
        if best_chord is None or len(best_cover) == 0:
            # Can't cover remaining, add a chord that covers at least one
            for k in available_chords:
                if k in selected:
                    continue
                cover = set()
                if k + 1 in remaining:
                    cover.add(k + 1)
                if N - k + 1 in remaining:
                    cover.add(N - k + 1)
                if len(cover) > 0:
                    best_chord = k
                    best_cover = cover
                    break
        
        if best_chord is None:
            break
        
        selected.add(best_chord)
        remaining -= best_cover
    
    # Now we have selected chords. Let's verify.
    can, missing = can_cover(selected)
    
    if not can:
        # Fallback to known good solution
        selected = set(range(2, 21))  # 19 chords
        can, missing = can_cover(selected)
        if not can:
            raise ValueError(f"Could not cover length {missing}")
    
    # Add chords to edges
    for k in sorted(selected):
        edges.append((0, k))
    
    # Verify uniqueness and simplicity
    edge_set = set()
    for u, v in edges:
        if u == v:
            raise ValueError("Self-loop")
        e = (min(u, v), max(u, v))
        if e in edge_set:
            raise ValueError(f"Duplicate edge {e}")
        edge_set.add(e)
    
    # Construct cycles for lengths 3..40
    cycles = []
    chord_list = sorted(selected)
    
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            # Find k such that k+1 == L or N-k+1 == L
            k1 = L - 1
            k2 = N - L + 1
            
            found = False
            if k1 in selected:
                # Cycle using chord (0, k1) and path 0->1->...->k1->0
                cycle = list(range(0, k1 + 1))
                found = True
            elif k2 in selected:
                # Cycle using chord (0, k2) and path 0->39->...->k2->0
                cycle = [0] + list(range(N - 1, k2 - 1, -1))
                found = True
            
            if not found:
                raise ValueError(f"Could not construct cycle of length {L}")
        
        # Verify cycle validity
        if len(cycle) != len(set(cycle)):
            raise ValueError(f"Cycle {L} has duplicate vertices")
        if len(cycle) != L:
            raise ValueError(f"Cycle {L} has wrong length {len(cycle)}")
        for i in range(L):
            u = cycle[i]
            v = cycle[(i + 1) % L]
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                raise ValueError(f"Edge {e} missing for cycle {L}")
        
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
