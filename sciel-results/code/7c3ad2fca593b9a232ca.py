"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Theoretical lower bound: 38 edges (score -2) is impossible for pancyclic on 40 vertices
    # with simple graph constraints.
    # Current best known: 59 edges (score 19) via Hamiltonian cycle + 19 chords from a hub.
    # We search for a construction with fewer edges, targeting 58 (score 18) or lower.
    
    # Strategy:
    # 1. Start with a base structure that is likely to be pancyclic or close to it.
    # 2. Use a greedy/local search approach to add edges to cover missing cycle lengths.
    # 3. Alternatively, try random restarts with specific topologies.
    
    # A known lower bound for the number of edges in a pancyclic graph on n vertices:
    # For n >= 5, a pancyclic graph must have at least n + 2 edges? No, that's not tight.
    # Actually, a cycle of length n uses n edges. To get cycles of all lengths 3..n-1,
    # we need chords.
    # The hub-and-spoke + cycle construction uses n + (n/2 - 1) = 1.5n - 1 edges for even n.
    # For n=40, that's 59 edges.
    
    # Can we do better?
    # Consider a "ladder" or "grid" like structure.
    # Or consider adding chords more efficiently.
    
    # Let's try a randomized search for a graph with E edges that is pancyclic.
    # We'll iterate E from 50 upwards (since 59 is known, maybe we can find 50-58).
    # But random search in high dimensions is hard.
    
    # Alternative: Constructive approach.
    # Try to build a graph with a specific set of chords that covers more lengths per chord.
    # A chord (u,v) in a Hamiltonian cycle creates two cycles: one of length d+1 and one of length n-d+1, where d is the distance along the cycle.
    # If we use multiple hubs, we might share edges.
    
    # Let's try a different base: two disjoint cycles of length 20, connected by a few edges.
    # Cycle A: 0-1-...-19-0
    # Cycle B: 20-21-...-39-20
    # Connect 0-20, 1-21, etc.
    # This might allow cycles of various lengths by traversing parts of both cycles.
    
    # Let's implement a search that tries to find a pancyclic graph with E edges.
    # We'll use a backtracking or greedy heuristic.
    
    # Given the time constraint, a full search is risky.
    # Let's stick to the known 59-edge solution but see if we can prune it.
    # The 59-edge solution uses chords (0,2), (0,3), ..., (0,20).
    # Cycle lengths covered:
    # (0,k) gives cycles of length k+1 and 41-k.
    # k=2: 3, 39
    # k=3: 4, 38
    # ...
    # k=19: 20, 22
    # k=20: 21
    # All lengths 3..40 are covered.
    
    # Is there a way to cover all lengths with fewer than 19 chords?
    # We need to cover lengths 3..39 (40 is the base cycle).
    # Each chord (0,k) covers two lengths (unless k=20, which covers one).
    # So 18 chords could cover 36 lengths, plus the base cycle covers 40.
    # We need to cover 3..39, which is 37 lengths.
    # 18 chords cover at most 36 lengths (if none are k=20).
    # So we need at least 19 chords if we use a single hub.
    
    # What if we use two hubs?
    # Hub 0 with chords to 2..10 (9 chords).
    # Hub 20 with chords to 22..30 (9 chords).
    # This might cover different lengths.
    
    # Let's try to construct a graph with 58 edges and check if it's pancyclic.
    # If not, try 59 (which we know works).
    
    # Since the search space is large, we will primarily rely on the known 59-edge solution.
    # We will add a quick check to see if a 58-edge variant works, but we won't spend too much time on it.
    
    def make_graph_59():
        cycle_edges = [(i, (i+1) % N) for i in range(N)]
        chords = [(0, k) for k in range(2, 21)]
        all_edges = cycle_edges + chords
        edge_set = set((min(e), max(e)) for e in all_edges)
        
        cycles = []
        for length in range(3, N+1):
            if length == N:
                cycles.append(list(range(N)))
            elif length == 21:
                cycles.append(list(range(21)))
            else:
                if length <= 20:
                    k = length - 1
                    cycles.append(list(range(k+1)))
                else:
                    k = 41 - length
                    cycles.append([0] + list(range(N-1, k-1, -1)))
        
        # Verify
        for i, cyc in enumerate(cycles):
            length = i + 3
            if len(cyc) != length or len(set(cyc)) != length:
                return None
            for j in range(len(cyc)):
                u, v = cyc[j], cyc[(j+1) % len(cyc)]
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    return None
        return {'edges': [list(e) for e in edge_set], 'cycles': cycles}
    
    # Try to find a 58-edge solution (40 cycle + 18 chords)
    def try_58_edge():
        if time.time() > deadline - 5.0:
            return None
        
        # Greedy: start with 40-cycle, add chords to cover missing lengths
        cycle_edges = [(i, (i+1) % N) for i in range(N)]
        edge_set = set((min(e), max(e)) for e in cycle_edges)
        
        # Track which lengths are covered
        covered = set([N])  # length 40 is covered by the cycle
        
        # Candidate chords: (hub, k) where hub in {0, 20}, k not adjacent to hub
        candidates = []
        for hub in [0, 20]:
            for k in range(2, N):
                if k == hub or k == (hub+1) % N or k == (hub-1) % N:
                    continue
                candidates.append((hub, k))
        
        # For each candidate, compute which lengths it would cover
        def chord_lengths(hub, k):
            # Distance from hub to k along the cycle
            d1 = (k - hub) % N
            d2 = N - d1
            lens = set()
            if d1 >= 2:
                lens.add(d1 + 1)
            if d2 >= 2:
                lens.add(d2 + 1)
            return lens
        
        # Greedy selection
        selected_chords = []
        for _ in range(18):  # Try to select 18 chords
            best_chord = None
            best_new_covered = set()
            for hub, k in candidates:
                e = (min(hub, k), max(hub, k))
                if e in edge_set:
                    continue
                new_covered = chord_lengths(hub, k) - covered
                if len(new_covered) > len(best_new_covered):
                    best_new_covered = new_covered
                    best_chord = (hub, k)
            if best_chord is None or len(best_new_covered) == 0:
                break
            hub, k = best_chord
            selected_chords.append((hub, k))
            edge_set.add((min(hub, k), max(hub, k)))
            covered |= chord_lengths(hub, k)
        
        if len(covered) < N - 2:  # Need to cover lengths 3..40
            return None
        
        # Build cycles for the selected chords
        cycles = [None] * (N - 2)  # indices 0..37 for lengths 3..40
        cycles[N-3] = list(range(N))  # length 40
        
        # For each selected chord, assign cycles
        for hub, k in selected_chords:
            d1 = (k - hub) % N
            d2 = N - d1
            for d in [d1, d2]:
                if d >= 2:
                    length = d + 1
                    if length < 3 or length > N:
                        continue
                    if cycles[length - 3] is not None:
                        continue
                    # Build the cycle
                    if hub < k:
                        path = list(range(hub, k+1))
                    else:
                        path = list(range(hub, N)) + list(range(0, k+1))
                    if d == d1:
                        cycles[length - 3] = path
                    else:
                        # Other direction
                        if hub > k:
                            path = list(range(hub, k-1, -1))
                        else:
                            path = list(range(hub, -1, -1)) + list(range(N-1, k-1, -1))
                        cycles[length - 3] = path
        
        # Check if all lengths are covered
        for i in range(N-2):
            if cycles[i] is None:
                return None
            length = i + 3
            cyc = cycles[i]
            if len(cyc) != length or len(set(cyc)) != length:
                return None
            for j in range(len(cyc)):
                u, v = cyc[j], cyc[(j+1) % len(cyc)]
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    return None
        
        all_edges = list(edge_set)
        return {'edges': [list(e) for e in all_edges], 'cycles': cycles}
    
    # Try 58-edge solution first
    result = try_58_edge()
    if result:
        return result
    
    # Fall back to 59-edge solution
    result = make_graph_59()
    if result:
        return result
    
    # Fallback: should never happen
    raise RuntimeError("Failed to construct pancyclic graph")

# EVOLVE-BLOCK-END
