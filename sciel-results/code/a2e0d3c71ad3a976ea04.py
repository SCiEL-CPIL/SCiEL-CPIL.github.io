"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices.
    Strategy: Start with a Hamiltonian cycle (40 edges), then greedily add chords
    to cover all cycle lengths 3..40. Minimize total edge count - 40.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # We need cycles of every length 3..40
    # A cycle of length k exists if we can find k distinct vertices forming a cycle in the graph
    
    # For a Hamiltonian cycle, any subset of consecutive vertices forms a path,
    # but we need the closing edge too. So for length k, we need k vertices where
    # the first and last are connected.
    
    # Key insight: In a pure cycle, only the full cycle (length 40) exists.
    # We need chords to create shorter cycles.
    
    # Greedy approach: For each required cycle length k (3 to 39), find the minimum
    # set of chords that enables a cycle of that length.
    
    # A chord (i,j) on the Hamiltonian cycle creates a cycle of length min(|i-j|, N-|i-j|)
    # when combined with the arc between i and j.
    
    # More precisely, chord (i,j) creates two cycles:
    # - Cycle of length (j-i) + 1 using arc i->i+1->...->j and chord j->i
    # - Cycle of length (N - (j-i)) + 1 using arc j->j+1->...->i (wrapping) and chord i->j
    
    # So each chord covers two cycle lengths: d and N-d where d = j-i (assuming j>i)
    
    # We need to cover all lengths 3..39 (length 40 is the Hamiltonian cycle itself)
    # Each chord (i,j) with d = j-i (1 < d < N-1) covers lengths d+1 and N-d+1... 
    # wait, let me reconsider.
    
    # Chord (i,j) where j > i: the two arcs have lengths (j-i) and (N-(j-i)) edges.
    # Adding the chord gives cycles of length (j-i)+1 and (N-(j-i))+1.
    # So chord with gap d covers cycle lengths d+1 and N-d+1.
    
    # We need to cover lengths 3, 4, ..., 39.
    # Length 3: need chord with d=2 (covers 3 and 39)
    # Length 4: need chord with d=3 (covers 4 and 38)
    # ...
    # Length k: need chord with d=k-1 (covers k and N-k+1)
    
    # So we need chords for d = 2, 3, ..., 37 to cover lengths 3..39.
    # But each chord covers TWO lengths: d+1 and N-d+1.
    # d=2 covers 3 and 39
    # d=3 covers 4 and 38
    # ...
    # d=19 covers 20 and 22
    # d=20 covers 21 and 21 (same!)
    # d=21 covers 22 and 20 (same as d=19)
    
    # So we need chords for d = 2, 3, ..., 20 (that's 19 chords) to cover all of 3..39.
    # Total edges: 40 (Hamiltonian) + 19 (chords) = 59. Score: 59-40 = 19.
    
    # But wait - we can potentially do better by using chords that create cycles
    # through combinations, not just single chords.
    
    # Let's start with the simple approach: 19 chords for gaps 2..20.
    # Place them strategically.
    
    chords = []
    for d in range(2, 21):  # gaps 2 through 20
        # Place chord (0, d) - all from vertex 0
        chords.append((0, d))
    
    all_edges = list(base_edges) + chords
    
    # Now we need to verify and construct explicit cycles for each length 3..40.
    # For length k (3 <= k <= 39):
    #   If k <= 21: use chord (0, k-1) and arc 0->1->...->(k-1)->0
    #     cycle: [0, 1, 2, ..., k-1]
    #   If k >= 21: use chord (0, N-k+1) and the long arc
    #     cycle: [0, N-k+1, N-k+2, ..., N-1, N-k, ..., 1] -- wait, this is getting complex
    
    # Let me think more carefully.
    # Chord (0, d) creates:
    #   Short cycle: 0->1->2->...->d->0 (length d+1)
    #   Long cycle: 0->d->d+1->...->N-1->N-2->...->1->0 ... no wait.
    #   The Hamiltonian cycle is 0->1->2->...->39->0.
    #   Chord (0,d) adds edge 0-d.
    #   Short cycle: 0,1,2,...,d (length d+1) - uses edges (0,1),(1,2),...,(d-1,d),(d,0)
    #   Long cycle: 0,d,d+1,...,39,38,...,1 ... no.
    #   Long cycle: 0, 39, 38, ..., d+1, d (length N-d+1) - uses edges (0,39),(39,38),...,(d+1,d),(d,0)
    #   Wait, that's not right either. The Hamiltonian cycle edges are (i, i+1) mod N.
    #   So from 0, we can go to 1 or to 39 (which is (0-1) mod N).
    #   Long cycle using chord (0,d): 0 -> 39 -> 38 -> ... -> d -> 0
    #     This uses edges (0,39), (39,38), ..., (d+1, d), (d, 0). Length = N - d + 1.
    
    # So for chord (0, d):
    #   Short cycle length: d + 1, vertices: [0, 1, 2, ..., d]
    #   Long cycle length: N - d + 1, vertices: [0, N-1, N-2, ..., d]
    
    # For d=2: short=3 [0,1,2], long=39 [0,39,38,...,2]
    # For d=3: short=4 [0,1,2,3], long=38 [0,39,38,...,3]
    # ...
    # For d=20: short=21 [0,1,...,20], long=21 [0,39,38,...,20]
    
    # So we cover:
    # Length 3: d=2 short
    # Length 4: d=3 short
    # ...
    # Length 21: d=20 short (or long)
    # Length 22: d=19 long (N-19+1=22)
    # Length 23: d=18 long
    # ...
    # Length 39: d=2 long
    
    # All lengths 3..39 are covered!
    
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        elif k <= 21:
            # Short cycle from chord (0, k-1)
            d = k - 1
            cycles.append(list(range(0, d + 1)))  # [0, 1, 2, ..., d]
        else:
            # Long cycle from chord (0, N-k+1)
            d = N - k + 1
            # Vertices: 0, N-1, N-2, ..., d
            cyc = [0] + list(range(N - 1, d - 1, -1))
            cycles.append(cyc)
    
    # Verify all edges in cycles exist
    edge_set = set(all_edges)
    for i, e in enumerate(all_edges):
        all_edges[i] = (min(e), max(e))
    edge_set = set(all_edges)
    
    valid = True
    for cyc in cycles:
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to seed
        return {'edges': [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)],
                'cycles': [list(range(k)) for k in range(3, N + 1)]}
    
    return {'edges': all_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
