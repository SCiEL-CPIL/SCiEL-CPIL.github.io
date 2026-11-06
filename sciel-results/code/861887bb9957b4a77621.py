"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40
DEADLINE = 90.0  # wall-clock seconds

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # Seed: elementary fan graph (star + path)
    seed_edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
    seed_edge_set = set(tuple(sorted(e)) for e in seed_edges)
    seed_cycles = [list(range(k)) for k in range(3, N+1)]
    seed_score = len(seed_edges) - N  # 39 + 38 - 40 = 37
    
    def is_valid_solution(edges, cycles):
        edge_set = set(tuple(sorted(e)) for e in edges)
        if len(edge_set) != len(edges):
            return False
        for i, v in enumerate(edges):
            if len(v) != 2 or v[0] < 0 or v[0] >= N or v[1] < 0 or v[1] >= N:
                return False
            if v[0] == v[1]:
                return False
        for k, cyc in enumerate(cycles):
            expected_len = k + 3
            if len(cyc) != expected_len:
                return False
            if len(set(cyc)) != expected_len:
                return False
            for j in range(expected_len):
                a, b = cyc[j], cyc[(j+1) % expected_len]
                if (a, b) in edge_set:
                    continue
                elif (b, a) in edge_set:
                    continue
                else:
                    return False
        return True
    
    # Check seed
    if not is_valid_solution(seed_edges, seed_cycles):
        return {'edges': seed_edges, 'cycles': seed_cycles}
    
    best_edges = seed_edges
    best_cycles = seed_cycles
    best_score = seed_score
    
    # Strategy: Chord-pair cycles
    # Base: Cycle C = (0, 1, 2, ..., 39). Edges: (i, i+1 mod 40) for i in 0..39.
    # Chords: pairs (u, v) that are NOT adjacent in C.
    # For each chord (u, v), we can form cycles of length |u-v|+1 and 40-|u-v|+1.
    # Actually, the two cycles formed by adding chord (u,v) to the Hamiltonian cycle are:
    #   Path 1: u -> u+1 -> ... -> v  (length (v-u) mod 40 + 1)
    #   Path 2: u -> u-1 -> ... -> v  (length 40 - (v-u mod 40) + 1)
    # We need to cover lengths 3..40.
    # The base cycle already gives length 40.
    # Length 3: requires a triangle, i.e., a chord (u, v) where |u-v| mod 40 == 2 (or 38).
    #   If chord is (0,2): cycles are 0-1-2-0 (len 3) and 0-39-38-...-2-0 (len 39).
    # So we need chords that cover the remaining lengths.
    
    # Build adjacency for the base cycle
    base_edges = set()
    for i in range(N):
        base_edges.add((i, (i+1) % N))
        base_edges.add(((i+1) % N, i))
    
    # Precompute all possible chord lengths
    # For a chord (u, v), let d = (v - u) % N.
    # The two cycles have lengths d+1 and N-d+1.
    # We need to cover lengths 3..39 (40 is already covered by the base cycle).
    # Note: if d=2, lengths are 3 and 39.
    # if d=3, lengths are 4 and 38.
    # if d=4, lengths are 5 and 37.
    # if d=5, lengths are 6 and 36.
    # if d=6, lengths are 7 and 35.
    # if d=7, lengths are 8 and 34.
    # if d=8, lengths are 9 and 33.
    # if d=9, lengths are 10 and 32.
    # if d=10, lengths are 11 and 31.
    # if d=11, lengths are 12 and 30.
    # if d=12, lengths are 13 and 29.
    # if d=13, lengths are 14 and 28.
    # if d=14, lengths are 15 and 27.
    # if d=15, lengths are 16 and 26.
    # if d=16, lengths are 17 and 25.
    # if d=17, lengths are 18 and 24.
    # if d=18, lengths are 19 and 23.
    # if d=19, lengths are 20 and 22.
    # if d=20, lengths are 21 and 21. (only one cycle length 21)
    # So we need to cover {3,4,...,39} using chords.
    # Each chord covers 2 lengths (or 1 if d=20).
    # We need at least 19 chords to cover 37 lengths if each covers 2, but some cover 1.
    # Actually, lengths 3..39 are 37 lengths.
    # Chords with d in 1..19 cover 2 lengths each. d=20 covers 1.
    # So we need at least 19 chords (if we use d=2..19, we cover 3..38 in 18 chords? No.)
    # d=2 -> {3, 39}
    # d=3 -> {4, 38}
    # d=4 -> {5, 37}
    # ...
    # d=19 -> {20, 21}
    # So d=2..19 (18 chords) cover lengths 3..21 and 39..22? No.
    # d=2 -> 3, 39
    # d=3 -> 4, 38
    # d=4 -> 5, 37
    # d=5 -> 6, 36
    # d=6 -> 7, 35
    # d=7 -> 8, 34
    # d=8 -> 9, 33
    # d=9 -> 10, 32
    # d=10 -> 11, 31
    # d=11 -> 12, 30
    # d=12 -> 13, 29
    # d=13 -> 14, 28
    # d=14 -> 15, 27
    # d=15 -> 16, 26
    # d=16 -> 17, 25
    # d=17 -> 18, 24
    # d=18 -> 19, 23
    # d=19 -> 20, 22
    # So 18 chords cover all lengths 3..39 except 21.
    # We need a chord for 21, which is d=20 (or 19? no, d=19 gives 20,22).
    # d=20 gives 21. So we need 19 chords total.
    # Total edges = 40 (base) + 19 (chords) = 59.
    # Score = 59 - 40 = 19.
    # This matches the seed score! So this is not an improvement yet.
    # But we can try to use fewer chords by not using the base cycle as a Hamiltonian cycle.
    # Or use a different base structure.
    
    # Let's try a different approach: use a sparse graph and search for cycles.
    # Start with a random sparse graph and add edges until all cycles are found.
    # This is complex. Instead, let's try to minimize chords by using a different base.
    
    # Alternative: Use a star graph + a few extra edges.
    # The seed uses 77 edges. The chord-pair approach uses 59.
    # Can we do better?
    # If we use a graph with minimum degree 2, we need at least 40 edges.
    # If we add a Hamiltonian cycle (40 edges), we have 40 edges.
    # We need 37 more cycles. Each chord adds at most 2 cycles.
    # So we need at least 19 chords. Total 59 edges.
    # Is it possible to cover 37 cycles with fewer than 19 chords?
    # No, because each chord adds at most 2 new cycle lengths.
    # Unless the cycles are not formed by a single chord on a Hamiltonian cycle.
    # For example, a cycle could be formed by two chords.
    # But that would likely require more edges.
    
    # Let's stick with the chord-pair approach and try to find a better base.
    # What if the base is not a single cycle?
    # For example, two disjoint cycles?
    # This might allow covering more lengths with fewer chords.
    
    # For now, let's implement the chord-pair search and see if we can find a valid solution with 59 edges.
    # Then we can try to optimize further.
    
    def find_chord_solution():
        """Find a solution using a Hamiltonian cycle + chords."""
        base_cycle = list(range(N))
        base_edges_set = set()
        for i in range(N):
            a, b = base_cycle[i], base_cycle[(i+1) % N]
            base_edges_set.add((min(a,b), max(a,b)))
        
        # We need to cover lengths 3..39.
        # Each chord (u,v) covers lengths |u-v|_cycle + 1 and N - |u-v|_cycle + 1.
        # Let's precompute the lengths covered by each possible chord.
        chord_lengths = {}
        for u in range(N):
            for v in range(u+1, N):
                d = (v - u) % N
                if d == 1 or d == N-1:
                    continue  # Already in base cycle
                l1 = d + 1
                l2 = N - d + 1
                chord_lengths[(u,v)] = {l1, l2}
        
        # Greedy set cover
        needed = set(range(3, N))  # 3..39
        chosen_chords = []
        remaining = set(needed)
        while remaining:
            best_chord = None
            best_cover = 0
            for chord, lengths in chord_lengths.items():
                cover = len(remaining & lengths)
                if cover > best_cover:
                    best_cover = cover
                    best_chord = chord
            if best_chord is None:
                break
            chosen_chords.append(best_chord)
            remaining -= chord_lengths[best_chord]
        
        # Build edges
        edges = list(base_edges_set)
        for chord in chosen_chords:
            edges.append(chord)
        
        # Build cycles
        cycles = [base_cycle]  # Length 40
        for chord in chosen_chords:
            u, v = chord
            # Find the two paths
            path1 = [u]
            cur = u
            while cur != v:
                cur = (cur + 1) % N
                path1.append(cur)
            cycles.append(path1)
            
            path2 = [u]
            cur = u
            while cur != v:
                cur = (cur - 1) % N
                path2.append(cur)
            cycles.append(path2)
        
        # We have cycles of various lengths. We need exactly one of each length 3..40.
        # Let's map length -> cycle
        length_to_cycle = {40: base_cycle}
        for cyc in cycles[1:]:
            l = len(cyc)
            if l not in length_to_cycle:
                length_to_cycle[l] = cyc
        
        # Check if all lengths 3..40 are covered
        for l in range(3, N+1):
            if l not in length_to_cycle:
                return None
        
        # Sort cycles by length
        final_cycles = [length_to_cycle[l] for l in range(3, N+1)]
        
        return edges, final_cycles
    
    sol = find_chord_solution()
    if sol:
        edges, cycles = sol
        if is_valid_solution(edges, cycles):
            score = len(edges) - N
            if score < best_score:
                best_edges = edges
                best_cycles = cycles
                best_score = score
    
    # Try random restarts with different base cycles
    random.seed(42)
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        # Random permutation as base cycle
        perm = list(range(N))
        random.shuffle(perm)
        base_cycle = perm
        base_edges_set = set()
        for i in range(N):
            a, b = base_cycle[i], base_cycle[(i+1) % N]
            base_edges_set.add((min(a,b), max(a,b)))
        
        # Greedy set cover on this base
        chord_lengths = {}
        for u in range(N):
            for v in range(u+1, N):
                # Check if (u,v) is in base cycle
                if (min(u,v), max(u,v)) in base_edges_set:
                    continue
                # Compute cycle distances
                # This is complex for arbitrary base. Skip for now.
                pass
        
        # Instead, let's try to use the known chord-pair structure but with different chords.
        # We'll use the base cycle 0..39 but try different chord sets.
        
    # Fallback to seed if no better solution found
    return {'edges': best_edges, 'cycles': best_cycles}

# EVOLVE-BLOCK-END
