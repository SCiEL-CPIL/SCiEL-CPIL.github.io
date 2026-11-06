import time
import random
from itertools import combinations

N = 40
DEADLINE = 80.0  # seconds, safety margin before 90s kill

def construct():
    """
    Constructs a sparse pancyclic graph on N=40 vertices.
    Uses a Hamiltonian cycle plus a minimum set cover of chords to create cycles of all lengths 3..39.
    The Hamiltonian cycle itself provides length 40.
    """
    start_time = time.time()
    
    # Base: Hamiltonian cycle on vertices 0..39
    # Edges: (i, i+1 mod N) for i in 0..39
    base_edges = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # We need to cover cycle lengths 3..39.
    # A chord (u, v) where u < v creates two new cycles:
    #   1. The "short" cycle along the path u -> u+1 -> ... -> v -> u. Length = (v - u) + 1.
    #   2. The "long" cycle along the path u -> u-1 -> ... -> v -> u. Length = (N - (v - u)) + 1 = N - v + u + 1.
    # Let d = v - u (distance along the cycle, 1..N-1). Since it's a chord, d != 1 and d != N-1 (those are base edges).
    # Actually, if d=1, it's a base edge. If d=N-1, it's a base edge.
    # So valid chords have d in 2..N-2.
    # The lengths covered are d+1 and N-d+1.
    # Note: d+1 + (N-d+1) = N+2.
    # We need to cover all L in 3..39.
    
    # Let's precompute for each possible chord (u, v) the set of lengths it covers.
    # To minimize edges, we want chords that cover "hard" lengths.
    # Note that length L and length N+2-L are paired.
    # If we cover L, we don't necessarily cover N+2-L with the same chord, but often they come in pairs or can be covered together.
    # Specifically, a chord with distance d covers L1 = d+1 and L2 = N-d+1.
    # L1 + L2 = N+2 = 42.
    # So chords cover pairs (L, 42-L).
    # The pairs in 3..39 are:
    # (3, 39), (4, 38), (5, 37), (6, 36), (7, 35), (8, 34), (9, 33), (10, 32), (11, 31), (12, 30), (13, 29), (14, 28), (15, 27), (16, 26), (17, 25), (18, 24), (19, 23), (20, 22), (21, 21).
    # Wait, if d+1 = 21, then N-d+1 = 40-20+1 = 21. So d=20 covers (21, 21).
    # So we need to cover all pairs. Each chord covers exactly one pair (or the same length twice if L=21).
    # Therefore, we need at least 18 chords to cover the 18 distinct pairs?
    # Let's check:
    # Pairs:
    # 3,39 -> d=2
    # 4,38 -> d=3
    # ...
    # 20,22 -> d=19
    # 21,21 -> d=20
    # So yes, each d from 2 to 20 covers a unique pair of lengths.
    # To cover all lengths 3..39, we must pick a set of chords such that for every pair (L, 42-L), at least one chord with the corresponding distance d is present.
    # Since each d covers a unique pair, and the pairs are disjoint (except for 21 which is its own pair), we MUST pick at least one chord for each d in 2..20?
    # No! A chord (u, v) has a specific distance d. But can a single chord cover two different pairs? No, because the lengths are determined solely by d.
    # So, to cover length 3, we need a chord with d=2.
    # To cover length 4, we need a chord with d=3.
    # ...
    # To cover length 20, we need a chord with d=19.
    # To cover length 21, we need a chord with d=20.
    # To cover length 22, we need a chord with d=18 (since 22 = 40-18+1).
    # To cover length 23, we need a chord with d=17.
    # ...
    # To cover length 39, we need a chord with d=2.
    #
    # So, if we pick a chord with d=2, it covers both 3 and 39.
    # If we pick a chord with d=3, it covers both 4 and 38.
    # ...
    # If we pick a chord with d=20, it covers 21.
    #
    # Thus, to cover ALL lengths 3..39, we MUST have:
    # - A chord with d=2 (covers 3, 39)
    # - A chord with d=3 (covers 4, 38)
    # - ...
    # - A chord with d=20 (covers 21)
    #
    # This implies we need at least 19 chords (d=2 to d=20).
    # The current solution uses exactly 19 chords.
    # Is it possible to use fewer?
    # Only if one chord could cover lengths from two different pairs. But that's impossible because each chord's lengths are fixed by its distance d.
    # Wait, is it possible that a cycle of length L is formed by a combination of chords?
    # The problem statement says "every required edge must exist". It doesn't say the cycle must be formed by just base edges + 1 chord.
    # However, if we use a more complex cycle, we might be able to "skip" a chord for a specific d.
    # But typically, in these sparse pancyclic constructions, the "shortest" cycle of length L is formed by a single chord.
    # If we don't have a chord with distance d, can we still form a cycle of length d+1?
    # A cycle of length L uses L edges. If we don't have the "shortest" chord, we might be able to form the cycle using a "longer" path and some other chords?
    # No, a simple cycle of length L in a graph with a Hamiltonian cycle + chords...
    # If we omit the chord with d=2, can we form a triangle (3-cycle)?
    # A triangle requires 3 edges. The base cycle has no triangles.
    # To get a triangle, we need a chord (u, v) such that the path from u to v is length 2 (i.e., d=2).
    # If we don't have any chord with d=2, we cannot form a triangle using just base edges and one chord.
    # Could we form a triangle using two chords?
    # Triangle vertices a, b, c.
    # Edges (a,b), (b,c), (c,a).
    # If (a,b) and (b,c) are base edges, then (c,a) must be a chord.
    # Distance between c and a in the base cycle:
    # If a=0, b=1, c=2. (0,1) base, (1,2) base. (2,0) is a chord with d=2.
    # If a=0, b=1, c=3. (0,1) base, (1,3) chord (d=2), (3,0) chord (d=3).
    # This uses two chords.
    # If we don't have d=2 chords, we can't have (1,3) or (2,0).
    # So we can't form a triangle.
    # Thus, we MUST have at least one chord with d=2.
    # Similarly, we MUST have at least one chord with d=3 to form a 4-cycle?
    # 4-cycle: 0-1-2-3-0. (0,1),(1,2),(2,3) base. (3,0) chord d=3.
    # Alternative: 0-1-3-2-0? (0,1) base, (1,3) chord d=2, (3,2) base, (2,0) chord d=2.
    # This uses two d=2 chords.
    # If we have d=2 chords, can we form a 4-cycle?
    # Yes, 0-1-3-2-0.
    # So we don't necessarily need a d=3 chord if we have two d=2 chords?
    # But we need to cover ALL lengths.
    # This suggests that the "one chord per distance" is a sufficient condition, but maybe not necessary if we allow complex cycles.
    # However, minimizing edge count is the goal.
    # 19 chords is the lower bound for the "simple" construction.
    # Can we do better with complex cycles?
    # Likely not, as the "complexity" adds edges or requires more chords to enable the paths.
    #
    # Let's stick with the 19 chords but try to optimize their placement.
    # Does placement matter for the edge count? No, we just need 19 chords.
    # Does placement matter for validity? No, any chord with the right distance works.
    #
    # Wait, the current solution uses chords (0, k) for k=2..20.
    # This creates a "fan" from vertex 0.
    # Is there a way to reduce edges?
    # The base cycle has 40 edges.
    # We add 19 chords. Total 59.
    # Value = 59 - 40 = 19.
    #
    # Is it possible to have fewer than 19 chords?
    # If we can cover multiple distances with one chord? No.
    # If we can form cycles without chords? No, base cycle only has length 40.
    #
    # What if we modify the base graph?
    # The base graph doesn't have to be a single cycle.
    # But a Hamiltonian cycle is the most efficient way to ensure connectivity and provide the "long" paths.
    #
    # Let's try to verify if 19 is indeed the minimum.
    # If we use a different base graph, say a path?
    # Path 0-1-...-39.
    # Cycles must close.
    # To make a cycle of length L, we need to connect two vertices in the path such that the segment has L-1 edges, plus the closing edge.
    # This is similar to the cycle case.
    #
    # Let's assume 19 chords is optimal for this structure.
    # Can we improve the score by reducing the number of edges further?
    # Maybe by sharing edges between cycles?
    # The edges are already shared.
    #
    # Let's try a different heuristic:
    # Instead of fixed chords, use a greedy set cover to find the minimum number of chords needed to cover all lengths, assuming complex cycles are allowed.
    # But as argued, simple cycles require specific distances.
    #
    # Let's just output the current best known construction (19 chords).
    # The current code does this.
    #
    # Is there any variation that could yield fewer edges?
    # What if we use a chord that is part of multiple cycles in a way that saves edges?
    # No, the edge count is just the number of distinct edges.
    #
    # Let's try to see if we can remove a base edge and add a chord to compensate?
    # If we remove a base edge, we break the Hamiltonian cycle.
    # We might lose the ability to form long cycles.
    #
    # Let's try a randomized search to see if a different set of 19 chords works (it will, all do).
    # Or maybe a set of 18 chords works with complex cycles?
    #
    # Let's try to construct a graph with 18 chords and check if it's pancyclic.
    # We will omit the chord for d=20 (length 21).
    # Can we form a 21-cycle without a d=20 chord?
    # A 21-cycle uses 21 edges.
    # If we use the base path of length 20 (21 vertices), we need to close it.
    # The closing edge must connect the ends.
    # If the path is 0..20, we need (20,0). This is d=20.
    # If we don't have (20,0), can we form a 21-cycle differently?
    # Maybe 0-1-...-19-21-20-0?
    # Edges: (0,1)..(19,20) are not all base. (19,21) is a chord?
    # This is getting complicated.
    #
    # Given the time limit, I will stick with the 19-chord solution but ensure it's robust.
    # The current solution is already 19 chords.
    #
    # Wait, the score is 1.9474.
    # Score = seed / value.
    # Seed = 19 (presumably, if value=19 gives score 1.0? No, seed program score 1.0).
    # If seed value is 19, and current value is 19, score should be 1.0.
    # But score is 1.9474.
    # 1.9474 * 19 = 37.0.
    # So seed value was 37?
    # 37 - 40 = -3? No.
    # Value = edges - 40.
    # If seed edges = 77, value = 37.
    # Current edges = 59, value = 19.
    # Score = 37 / 19 = 1.947.
    #
    # So we need to reduce edges further.
    # Can we go below 59 edges?
    # i.e., fewer than 19 chords?
    #
    # Let's try to find a graph with 18 chords.
    # We need to cover all lengths 3..39.
    # If we omit d=20, we need to form a 21-cycle.
    #
    # Let's try a greedy approach to find the minimum chords.
    
    # Universe: lengths 3..39
    # Sets: for each possible chord (u,v), the lengths it can form.
    # For simplicity, assume only simple cycles (base path + 1 chord).
    # Then each chord covers exactly 2 lengths (or 1 if d=20).
    # And these pairs are disjoint.
    # So we MUST have 19 chords.
    #
    # To beat 19, we MUST use complex cycles.
    #
    # Let's try to build a graph with 18 chords and see if we can find cycles for all lengths.
    # We will omit the chord for d=20.
    # Chords: d=2..19.
    # We have chords for d=2, 3, ..., 19.
    # Lengths covered directly:
    # d=2 -> 3, 39
    # d=3 -> 4, 38
    # ...
    # d=19 -> 20, 22
    # Missing: 21.
    #
    # Can we form a 21-cycle?
    # We have chords for d=2..19.
    # Try to combine them.
    # A 21-cycle has 21 edges.
    #
    # Let's try: 0-1-...-10-12-11-13-...-20-0?
    # This is hard to construct manually.
    #
    # Let's write a small search for a 21-cycle in a graph with chords d=2..19.
    # Vertices 0..39.
    # Chords: (0,2), (0,3), ..., (0,19) ? No, we can place them anywhere.
    # To maximize flexibility, let's place chords as (0, d) for d=2..19.
    # So we have a fan from 0 to 2..19.
    # Plus base cycle.
    #
    # Can we find a 21-cycle?
    # Path from 0 to some vertex, then back to 0.
    # If we go 0 -> 19 -> 18 -> ... -> 0?
    # 0-19 is a chord.
    # 19-18 ... 1-0 are base edges.
    # Path: 0, 19, 18, ..., 1.
    # Length: 1 (chord) + 19 (base) = 20 edges. 21 vertices.
    # Cycle: 0-19-18-...-1-0.
    # This is a 21-cycle!
    #
    # Let's verify:
    # Vertices: 0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.
    # Edges:
    # (0,19) - Chord d=19. Exists.
    # (19,18) - Base.
    # ...
    # (2,1) - Base.
    # (1,0) - Base.
    #
    # So we CAN form a 21-cycle without a d=20 chord!
    #
    # So we only need chords for d=2..19.
    # That's 18 chords.
    #
    # Let's check if any other lengths are missing.
    # d=2 covers 3, 39.
    # d=3 covers 4, 38.
    # ...
    # d=19 covers 20, 22.
    # 21 is covered by the complex cycle.
    #
    # So 18 chords are sufficient!
    #
    # Can we go lower?
    # Omit d=19?
    # Then we need to cover 20 and 22.
    # 20-cycle:
    # Without d=19, can we form a 20-cycle?
    # Similar to 21-cycle, maybe 0-18-17-...-1-0?
    # 0-18 is chord d=18.
    # 18-17...1-0 base.
    # Length: 1 + 18 = 19 edges. 20 vertices.
    # So 20-cycle is formed by d=18 chord.
    #
    # 22-cycle:
    # Without d=19, can we form a 22-cycle?
    # 0-18-17-...-0 is 20-cycle.
    # We need 22.
    # Maybe 0-18-17-...-2-1-0? No, that's shorter.
    # Maybe use two chords?
    #
    # If we omit d=19, we lose direct 20 and 22.
    # But we found 20 is covered by d=18 (complex cycle 0-18-...-1-0 is length 20? Wait.)
    # Cycle 0-18-17-...-1-0.
    # Vertices: 0, 18, 17, ..., 1.
    # Count: 1 (0) + 18 (18..1) = 19 vertices?
    # No, 18 down to 1 is 18 vertices. Plus 0 is 19 vertices.
    # Edges: (0,18) + (18,17)..(1,0).
    # Base edges: 18-17, ..., 1-0. That's 18 edges.
    # Total edges: 19.
    # So it's a 19-cycle.
    #
    # My previous calculation for 21-cycle:
    # 0-19-18-...-1-0.
    # Vertices: 0, 19, 18, ..., 1.
    # 19 down to 1 is 19 vertices. Plus 0 is 20 vertices.
    # Edges: (0,19) + 19 base edges.
    # Total 20 edges.
    # So it's a 20-cycle.
    #
    # So d=19 chord allows forming a 20-cycle (complex).
    # And d=19 chord directly forms 20-cycle (short) and 22-cycle (long).
    #
    # If we omit d=19:
    # We have d=2..18.
    # Directly covered:
    # 3,39 (d=2)
    # ...
    # 19,23 (d=18)
    #
    # Missing: 20, 21, 22.
    #
    # Can we form 20, 21, 22 with d=2..18?
    #
    # 20-cycle:
    # We have d=18.
    # Complex cycle using d=18:
    # 0-18-17-...-1-0 is 19-cycle.
    # 0-18-17-...-0 (short) is 19-cycle.
    # 0-18-17-...-39-0 (long) is 23-cycle.
    #
    # We need 20.
    # Maybe 0-18-17-...-2-0?
    # 0-18 chord. 18-17...2-1 base. 1-0 base.
    # Vertices: 0, 18, ..., 1. (19 vertices).
    #
    # Maybe use two chords?
    # 0-18-...-x-y-...-0.
    #
    # This is getting risky.
    # 18 chords seems safe.
    #
    # Let's implement 18 chords (d=2..19).
    # And verify that 21 is covered.
    
    # Chords: (0, k) for k in 2..19.
    # Total edges: 40 + 18 = 58.
    # Value: 58 - 40 = 18.
    # Score: 37 / 18 = 2.055.
    
    # Let's double check the 21-cycle with chords d=2..19.
    # We need a cycle of length 21.
    # Proposed: 0-19-18-...-1-0.
    # Edges: (0,19) [chord], (19,18)..(1,0) [base].
    # Count: 1 + 19 = 20 edges.
    # Vertices: 0, 19, 18, ..., 1.
    # Count: 1 + 19 = 20 vertices.
    # This is a 20-cycle.
    
    # We need a 21-cycle.
    # Try: 0-19-18-...-0? No.
    # Try: 0-19-18-...-2-1-0?
    # 0-19 chord.
    # 19-18...1-0 base.
    # This is the same 20-cycle.
    
    # We need to include one more vertex.
    # Maybe 0-19-18-...-1-39-0?
    # 0-19 chord.
    # 19-18...1-0 base? No, 1-39 is base. 39-0 is base.
    # Path: 0-19-18-...-1-39-0.
    # Edges:
    # (0,19) chord.
    # (19,18)..(2,1) base.
    # (1,39) base.
    # (39,0) base.
    #
    # Vertices: 0, 19, 18, ..., 1, 39.
    # Count: 1 (0) + 19 (19..1) + 1 (39) = 21 vertices.
    # Edges:
    # (0,19)
    # (19,18)
    # ...
    # (2,1)
    # (1,39)
    # (39,0)
    #
    # Total edges: 1 + 18 + 1 + 1 = 21 edges.
    #
    # Is (1,39) a base edge?
    # Base cycle: 0-1-2-...-39-0.
    # (1,39) is NOT a base edge. (1,2) and (39,0) are.
    # (1,39) is a chord with d=2 (distance 39-1=38, or 1+40-39=2).
    # We have chord (0,2) for d=2. Do we have (1,39)?
    # We placed chords as (0,k). So we have (0,2).
    # We do NOT have (1,39).
    #
    # So this cycle is invalid.
    
    # We need to form a 21-cycle using only base edges and chords (0,k) for k=2..19.
    #
    # Let's try: 0-19-18-...-1-0 is 20-cycle.
    # To make it 21, we need to detour.
    #
    # What if we use a different chord?
    # We have (0,2).
    # Cycle: 0-2-1-0? Triangle.
    #
    # Let's try to find a 21-cycle programmatically.
    
    def find_cycle(edges, length):
        # Simple DFS to find a cycle of given length
        # This is exponential, so only for small N or limited depth
        pass

    # Given the complexity, let's stick to 19 chords if we can't easily prove 18 works.
    # But wait, if I can't find the cycle, I can't output it.
    # The task requires outputting the cycles.
    
    # Let's try to construct the 21-cycle for the 18-chord case (chords 0-2..0-19).
    # We need 21 distinct vertices.
    #
    # How about: 0-19-18-...-10-11-12-...-1-0?
    # No, that's not a simple path.
    
    # Let's try: 0-19-18-...-1-39-38-...-20-0?
    # 0-19 chord.
    # 19-18...1-0 base? No, 1-0 is base.
    # But we need to close to 0.
    #
    # Let's try: 0-19-18-...-1-39-0.
    # Edges:
    # (0,19) chord.
    # (19,18)..(1,0) base?
    # (1,0) is base.
    # (1,39) is NOT base.
    #
    # So we need (1,39) or similar.
    # We don't have it.
    
    # It seems 18 chords might not work with this specific placement.
    # What if we place chords differently?
    #
    # Let's go back to 19 chords. It's safe and known to work.
    # Score 1.9474 is good.
    #
    # Can we improve by using a different base graph?
    # No, 19 chords is likely optimal for this family.
    
    # Let's just return the 19-chord solution.
    
    # To ensure robustness, I will use the 19-chord solution.
    
    chords = []
    for k in range(2, 21):  # 2..20
        edges = base_edges | set([(0, k) if 0 < k else (k, 0) for k in [k]])
        # Actually just add to set
        pass
    
    final_edges = set(base_edges)
    for k in range(2, 21):
        final_edges.add((0, k))
    
    edge_list = [tuple(sorted(e)) for e in final_edges]
    
    # Construct cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L <= 21:
            k = L - 1
            cyc = list(range(k + 1))
        else:
            k = N + 1 - L
            cyc = [0] + list(range(N - 1, k - 1, -1))
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == '__main__':
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    for cyc in res['cycles']:
        L = len(cyc)
        if L < 3 or L > 40:
            raise ValueError(f"Invalid cycle length {L}")
        if len(set(cyc)) != L:
            raise ValueError(f"Duplicate vertices in cycle of length {L}")
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            if (u, v) not in edges and (v, u) not in edges:
                raise ValueError(f"Missing edge ({u},{v}) in cycle of length {L}")
    print(f"Edges: {len(res['edges'])}, Cycles: {len(res['cycles'])}")
