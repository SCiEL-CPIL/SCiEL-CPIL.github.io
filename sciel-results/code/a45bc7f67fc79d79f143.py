"""Sparse pancyclic graphs on 40 vertices: Hamiltonian cycle + minimal chords via search."""
import time
import random
from itertools import combinations, permutations

N = 40

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 75.0  # 75s out of 90s budget

    # Base: Hamiltonian cycle 0-1-2-...-39-0 (40 edges)
    # Cycle for length 40 is the Hamiltonian cycle itself.
    # For length k (3 <= k <= 39), we need a k-cycle using a subset of vertices.
    # The Hamiltonian cycle gives us a cycle of length 40.
    # For shorter cycles, we need chords.

    # Strategy: Use the Hamiltonian cycle as base.
    # For a cycle of length k, we can take k consecutive vertices on the Hamiltonian cycle
    # and add one chord to close the cycle. But that only works if k consecutive vertices
    # form a path of length k-1, and we need one more edge to close it.
    # Actually, k consecutive vertices on the Hamiltonian cycle form a path of k-1 edges.
    # To make a cycle of length k, we need one additional edge connecting the two endpoints.
    # So for each k from 3 to 39, we need edge (v, v+k-1) mod 40 for some starting vertex v.
    # But we can choose v to minimize total distinct chords needed.

    # Let's think: for cycle of length k starting at vertex i:
    # vertices: i, i+1, ..., i+k-1 (mod 40)
    # edges: (i,i+1), (i+1,i+2), ..., (i+k-2,i+k-1) -- these are all on the Hamiltonian cycle
    # plus chord: (i, i+k-1) mod 40
    # So we need chord (i, (i+k-1) mod 40) for each k.

    # We want to minimize the number of distinct chords.
    # For each k in 3..39, we need a chord of the form (i, (i+k-1)%40) for some i.
    # The chord is determined by the pair {i, (i+k-1)%40}.
    # Note that (i+k-1)%40 - i = k-1 (mod 40). So the chord has "gap" k-1 mod 40.
    # For a given gap d = k-1, the chord connects vertices that are d apart on the cycle.
    # There are 40 such chords for each d (one for each starting vertex), but they come in pairs:
    # chord (i, i+d) is the same as chord (i+d, i), and also chord (i, i+d) = chord (i+d, i).
    # Actually, for a fixed d, the set of chords {(i, (i+d)%40) : i=0..39} has 40/d' distinct edges
    # where d' = gcd(d, 40). Wait, no. Each pair {i, (i+d)%40} is an undirected edge.
    # The number of distinct undirected edges with gap d is 40/gcd(d,40) if d != 0, but actually:
    # For d and 40-d, the chords are the same set. So gap d and gap 40-d give the same chords.
    # For a fixed d (1 <= d <= 19, since d and 40-d are symmetric), the number of distinct chords
    # with gap d is 40/gcd(d,40).

    # We need one chord for each k in 3..39, i.e., for each gap d in 2..38.
    # But gap d and gap 40-d are the same. So we need chords for gaps 2,3,...,19 (since 40-d for d=2..19 gives 38..21, which are covered).
    # Wait: k ranges from 3 to 39, so d = k-1 ranges from 2 to 38.
    # Gap d and gap 40-d are the same set of chords. So we need chords for gaps 2,3,...,19.
    # For each gap d in 2..19, we need at least one chord with that gap.
    # A chord with gap d connects vertices i and (i+d)%40.

    # To minimize total chords, we want to choose one chord per gap, but a single chord can serve
    # multiple gaps? No, a chord has a specific gap. So we need at least one chord per gap d in 2..19.
    # That's 18 chords minimum. Plus 40 cycle edges = 58 edges total.

    # But wait: can a single chord serve two different k values? A chord (u,v) has gap d = |u-v| mod 40 (taking min(d, 40-d)).
    # It can only serve the k value where k-1 = d or k-1 = 40-d, i.e., k = d+1 or k = 41-d.
    # So a chord with gap d serves both k = d+1 and k = 41-d (if both are in 3..39).
    # For d in 2..19: k = d+1 is in 3..20, and k = 41-d is in 22..39.
    # So one chord with gap d can serve two k values: k=d+1 and k=41-d.
    # But we still need a chord for each d in 2..19, so 18 chords.

    # However, we can be smarter: for a given gap d, we can choose WHICH chord to use.
    # The choice affects whether the same chord can be used for other purposes, but since each chord
    # has a unique gap, we just need one per gap. So 18 chords + 40 = 58 edges.

    # Let's verify: can we do better? The lower bound is that we need a cycle of each length 3..40.
    # The Hamiltonian cycle handles 40. For each other length, we need at least one non-cycle edge
    # (chord). But one chord can help form multiple cycles. However, each chord has a fixed gap,
    # so it can only close cycles of specific lengths. I think 18 chords is optimal.

    # Let's just use one chord per gap d in 2..19, choosing chord (0, d) for each d.
    # Chords: (0,2), (0,3), ..., (0,19). That's 18 chords.
    # Total edges: 40 + 18 = 58.

    # Now let's verify that cycles exist:
    # For k = d+1 (d in 2..19): cycle is 0, 1, 2, ..., d. Edges: (0,1),(1,2),...,(d-1,d) [on cycle] + (0,d) [chord]. Length = d+1 = k. Good.
    # For k = 41-d (d in 2..19): cycle is 0, 39, 38, ..., 40-d. Wait, let me think again.
    # Vertices: 0, 39, 38, ..., (40-d)%40. The path goes 0 -> 39 -> 38 -> ... -> (40-d).
    # Number of vertices: from 0 down to (40-d), that's d+1 vertices? No.
    # 0, 39, 38, ..., 40-d. The count is: 0 is 1 vertex, then 39 down to 40-d is (39 - (40-d) + 1) = d vertices. Total = d+1 vertices.
    # But k = 41-d, so we need 41-d vertices. d+1 != 41-d unless d=20. So this doesn't work directly.

    # Let me reconsider. For gap d, the chord (0, d) connects 0 and d.
    # Cycle using this chord going forward: 0, 1, 2, ..., d. Length d+1.
    # Cycle using this chord going backward: 0, 39, 38, ..., d. Length = 40 - d + 1 = 41 - d.
    # Vertices: 0, 39, 38, ..., d. Count: from 39 down to d is (39-d+1) = 40-d vertices, plus 0 = 41-d vertices. Yes!
    # So chord (0,d) gives cycles of length d+1 and 41-d.
    # For d in 2..19: d+1 in 3..20, 41-d in 22..39.
    # We're missing k=21! For k=21, d=20 or d=20 (since 41-20=21). Gap 20.
    # Chord (0,20) gives cycles of length 21 and 21 (same). So we need chord (0,20) for k=21.
    # So we need chords for d in 2..20, that's 19 chords. Total edges: 40+19=59.

    # Hmm, the measured result was 59 edges. Let me check if we can do 58.
    # For k=21, we need a cycle of length 21. The gap would be 20 (or 20). So we need a chord with gap 20.
    # Gap 20 chords: (i, i+20) for i=0..19. These are 20 distinct chords (since gcd(20,40)=20, 40/20=2... wait).
    # Actually, (i, i+20) and (i+20, i+40%40=i) are the same. So there are 20 distinct chords with gap 20.
    # We need at least one. So 19 chords (d=2..20) is the minimum? Let's see if d=20 can be shared.
    # No, each gap needs its own chord. So 19 chords minimum.

    # Let's go with 59 edges: Hamiltonian cycle + chords (0,d) for d in 2..20.

    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        edges.add((i, (i+1) % N))

    # Chords from vertex 0
    for d in range(2, 21):
        edges.add((0, d))

    edges = list(edges)
    # Sort for determinism
    edges = sorted([(min(e), max(e)) for e in edges])

    # Now construct cycles
    cycles = []
    for k in range(3, N+1):
        if k == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            d = k - 1
            if d <= 20:
                # Forward: 0, 1, 2, ..., d
                cyc = list(range(d+1))
            else:
                # d > 20, so k > 21, use backward direction
                # k = 41 - d', where d' = 41 - k
                # Chord (0, d') where d' = 41-k
                dprime = 41 - k
                # Backward: 0, 39, 38, ..., dprime
                cyc = [0] + list(range(39, dprime-1, -1))
            cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
