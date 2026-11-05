"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # The current seed is a fan graph with 2N-3 = 77 edges.
    # Strategy: Try to find a graph with fewer edges (e.g., 76, 75, ...) 
    # that is still pancyclic.
    # 
    # A known sparse pancyclic graph family is the "ladder" or "grid" based, 
    # but for N=40, a "path with chords" is promising.
    #
    # Let's try a specific construction:
    # 1. Start with a Hamiltonian path 0-1-...-39 (39 edges).
    # 2. We need to add edges to create cycles of length 3..40.
    #    The path itself creates no cycles.
    #    Adding an edge (u, v) with v > u creates a cycle of length v - u + 1 
    #    using the path segment u...v.
    #    It also potentially creates other cycles if combined with other chords.
    #
    # If we only add chords (u, v) such that the path segment is the *only* way 
    # to connect them in the cycle, then each chord gives exactly one cycle length 
    # L = v - u + 1.
    # To get all lengths 3..40, we need chords for each length L.
    # For length L, we need a chord (u, u+L-1).
    # If we pick u=0 for all, we get chords (0, L-1).
    # Chords: (0,2), (0,3), ..., (0,39).
    # Number of chords: 38.
    # Total edges: 39 (path) + 38 (chords) = 77.
    # This is the seed again (fan at 0, but path is 0-1-...-39, chords 0-i).
    # Wait, the seed edges were: (0,i) for i=1..39 and (i,i+1) for i=1..38.
    # My path is 0-1-...-39. Edges (i, i+1) for i=0..38.
    # Chords (0, L-1) for L=3..40 -> (0,2)..(0,39).
    # Edge (0,1) is in path. (0,39) is a chord.
    # Total edges: Path (39) + Chords (37, since 0-1 is in path, 0-39 is new? No, 0-39 is not in path 0-1-...-39? Yes it is not adjacent in path).
    # Path edges: (0,1), (1,2), ..., (38,39). Count 39.
    # Chords needed for L=3: (0,2). L=4: (0,3). ... L=40: (0,39).
    # Chords: (0,2), (0,3), ..., (0,39). Count 38.
    # Total: 39 + 38 = 77.
    #
    # Can we do better?
    # If a single chord creates multiple cycles (by interacting with other chords), 
    # we might save edges.
    # However, finding such a configuration is a hard combinatorial problem.
    #
    # Alternative: Use a "cycle" as the backbone instead of a path.
    # Backbone: 0-1-...-39-0 (40 edges).
    # This gives cycle length 40.
    # We need lengths 3..39.
    # Adding a chord (u, v) splits the cycle into two cycles of lengths L1 and L2 
    # such that L1 + L2 = N + 2? No.
    # If we have a cycle of length N, and add a chord (u, v), we create two cycles.
    # Let the distance along the cycle be d and N-d.
    # The new cycles have lengths d+1 and N-d+1.
    # So one chord gives two cycle lengths.
    # We need to cover 3..39.
    # If we add chords strategically, we might cover all lengths with fewer chords.
    #
    # Let's try to construct a graph with a Hamiltonian cycle and ~N/2 chords.
    # Total edges ~ 40 + 20 = 60.
    #
    # Construction:
    # Cycle: (0, 1, ..., 39, 0).
    # Add chords (0, k) for k in some set.
    # Chord (0, k) creates cycles of length k+1 and 41-k.
    # We want to cover lengths 3..39.
    # Length 3: need chord (0,2) -> cycles 3 and 39.
    # Length 4: need chord (0,3) -> cycles 4 and 38.
    # ...
    # Length 20: need chord (0,19) -> cycles 20 and 21.
    # Length 21: covered by chord (0,19).
    # ...
    # Length 38: covered by chord (0,3).
    # Length 39: covered by chord (0,2).
    #
    # If we add chords (0, 2), (0, 3), ..., (0, 20):
    # Chords: 19 chords.
    # Cycles generated:
    # (0,2) -> 3, 39
    # (0,3) -> 4, 38
    # (0,4) -> 5, 37
    # ...
    # (0,19) -> 20, 21
    # (0,20) -> 21, 20
    #
    # Covered lengths: 3, 4, ..., 20, 21, ..., 39.
    # All lengths 3..39 are covered.
    # Plus the base cycle length 40.
    #
    # Total edges:
    # Base cycle: 40.
    # Chords: 19.
    # Total: 59.
    #
    # This is significantly better than 77!
    # Score improvement: 77/59 ~ 1.30.
    #
    # Let's verify this construction.
    # Vertices 0..39.
    # Edges:
    # 1. (i, (i+1)%N) for i in 0..N-1. (40 edges)
    # 2. (0, k) for k in 2..20. (19 edges)
    #
    # Check cycles:
    # For k in 2..20:
    #   Cycle A: 0 -> 1 -> ... -> k -> 0. Length k+1.
    #   Cycle B: 0 -> k -> k+1 -> ... -> 39 -> 0. Length N - k + 1.
    #
    # Set of lengths from chords:
    # k=2: 3, 39
    # k=3: 4, 38
    # k=4: 5, 37
    # k=5: 6, 36
    # k=6: 7, 35
    # k=7: 8, 34
    # k=8: 9, 33
    # k=9: 10, 32
    # k=10: 11, 31
    # k=11: 12, 30
    # k=12: 13, 29
    # k=13: 14, 28
    # k=14: 15, 27
    # k=15: 16, 26
    # k=16: 17, 25
    # k=17: 18, 24
    # k=18: 19, 23
    # k=19: 20, 22
    # k=20: 21, 21
    #
    # Union of lengths: {3, 4, ..., 20, 21, 22, ..., 39}.
    # Missing? 
    # 3..20 are covered by Cycle A.
    # 21 is covered by Cycle B of k=19 or k=20.
    # 22 is covered by Cycle B of k=18.
    # ...
    # 39 is covered by Cycle B of k=2.
    #
    # So lengths 3..39 are all present.
    # Length 40 is the base cycle 0-1-...-39-0.
    #
    # Total edges: 40 + 19 = 59.
    #
    # We need to return the specific cycles for each length 3..40.
    #
    # Let's build the edge list and cycle list.
    
    edges = []
    # Base cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords
    for k in range(2, 21):
        edges.append((0, k))
        
    # Remove duplicates and sort for consistency (though set is used internally)
    edges = list(set(edges))
    
    # Construct cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Base cycle
            cycles.append(list(range(N)))
        else:
            # Find k such that L = k + 1 or L = N - k + 1
            # Case 1: L = k + 1 => k = L - 1
            # Case 2: L = N - k + 1 => k = N - L + 1
            
            found = False
            # Try Case 1
            k1 = L - 1
            if 2 <= k1 <= 20:
                # Cycle: 0, 1, ..., k1, 0
                cycle = list(range(0, k1 + 1))
                # Verify edges exist
                # (0,1), (1,2)...(k1-1, k1), (k1, 0)
                # (k1, 0) is a chord if k1>=2. (0,1) is base.
                # All path edges exist.
                cycles.append(cycle)
                found = True
            
            if not found:
                # Try Case 2
                k2 = N - L + 1
                if 2 <= k2 <= 20:
                    # Cycle: 0, k2, k2+1, ..., N-1, 0
                    cycle = [0] + list(range(k2, N))
                    # Verify edges:
                    # (0, k2) chord.
                    # (k2, k2+1)...(N-2, N-1) base.
                    # (N-1, 0) base.
                    cycles.append(cycle)
                    found = True
            
            if not found:
                # This should not happen based on analysis, but fallback
                # If L is not covered by this specific set of chords, 
                # we might need to add more chords or change strategy.
                # For now, raise an error to detect failure.
                raise ValueError(f"Could not find cycle for length {L}")
        
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
