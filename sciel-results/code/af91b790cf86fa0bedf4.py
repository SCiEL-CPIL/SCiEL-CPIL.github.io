"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    def norm(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def build_adjacency(edges):
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # ---------- Fixed Hamiltonian cycle backbone ----------
    # Vertices 0..39 in order, edge (i, (i+1)%N)
    H = [(i, (i + 1) % N) for i in range(N)]

    # Precompute tree distances on the cycle (for each pair, the two arc lengths)
    # dist[i][j] = min(|i-j|, N-|i-j|)
    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            dist[i][j] = min(d, N - d)

    # For a chord (u,v) added to the cycle, it creates exactly two cycles:
    #   cycle A: length = dist[u][v] + 1  (shorter arc + chord)
    #   cycle B: length = (N - dist[u][v]) + 1  (longer arc + chord)
    # Both are valid simple cycles (arc is a path, plus the chord closes it).
    # So each chord covers two cycle lengths.

    # We need cycles of length 3..39 (length 40 = the Hamiltonian cycle itself).
    # For each chord (u,v), the two covered lengths are:
    #   l1 = dist[u][v] + 1,  l2 = N - dist[u][v] + 1
    # Note l1 + l2 = N + 2 = 42.
    # dist ranges 1..20 (but dist=1 means adjacent on cycle -> not a new edge)
    # So dist in 2..20 (dist=20 means antipodal, l1=l2=21)
    # l1 in 3..21, l2 in 21..39
    # We need to cover lengths 3..39.

    # Strategy: greedily pick chords to cover all lengths 3..39 with minimum chords.
    # Each chord covers {l1, l2} where l1+l2=42, l1 in 3..21.
    # Length 21 is covered by antipodal chord (dist=20): l1=l2=21.
    # Lengths 3..20: need l1=L, so dist=L-1.
    # Lengths 22..39: need l2=L, so dist=42-L-1... wait l2=N-dist+1=40-dist+1=41-dist.
    #   l2=L => dist=41-L. For L=22, dist=19; L=39, dist=2.
    # So length L in 3..20 needs a chord with dist=L-1 (in 2..19).
    # Length L in 21..39 needs a chord with dist=41-L (in 2..20).
    # Length 21 needs dist=20.

    # To minimize chords, each chord covers two lengths. Let's find a minimum set.
    # Lengths 3..39 = 37 lengths. Each chord covers 2 (or 1 if l1=l2=21).
    # So minimum ~19 chords (one covers 21 only, 18 others cover pairs).
    # Total edges = 40 (cycle) + 19 = 59. Hmm, same as before.

    # Wait, can we do better? The issue is each chord only covers 2 specific lengths.
    # But a vertex in the graph can be part of MANY cycles. The chord just needs to
    # EXIST; the cycle of length L doesn't have to be the "chord+arc" cycle.
    # The chord+arc gives us a CERTAIN cycle of length L, but other cycles may also exist.
    # However, for the certificate, we just need to exhibit one cycle per length.

    # Actually, the key insight: we don't need a separate chord per length!
    # A single chord (u,v) with dist=d creates cycles of length d+1 AND 41-d.
    # But if we have multiple chords, we can form cycles using MULTIPLE chords.
    # E.g., a cycle could go along the Hamiltonian cycle for some segments and use
    # chords for others, creating lengths not directly given by a single chord.

    # This makes the problem much richer. Let's use a greedy construction:
    # Start with the Hamiltonian cycle (40 edges). For each length L=3..39,
    # if no cycle of length L is yet certified, add the cheapest chord that creates one.
    # A chord (u,v) creates cycle length dist[u][v]+1 and 41-dist[u][v].
    # But also, combinations of chords can create other lengths.

    # Simpler robust approach: add chords to directly cover all lengths.
    # For L=3: need dist=2. Chord (0,2): cycle 0-1-2-0 (length 3) and 0-39-38-...-2-0 (length 39).
    #   Covers lengths 3 and 39.
    # For L=4: need dist=3. Chord (0,3): cycle 0-1-2-3-0 (length 4) and length 38.
    #   Covers 4 and 38.
    # ...
    # For L=20: need dist=19. Chord (0,19): covers 20 and 22.
    # For L=21: need dist=20. Chord (0,20): covers 21 and 21.
    # For L=22: need dist=19. Already covered by chord (0,19)?
    #   Chord (0,19) covers 20 and 22. Yes!
    # So let's pair them up:
    #   L=3  -> dist=2  -> chord (0,2)  -> also covers 39
    #   L=4  -> dist=3  -> chord (0,3)  -> also covers 38
    #   L=5  -> dist=4  -> chord (0,4)  -> also covers 37
    #   ...
    #   L=20 -> dist=19 -> chord (0,19) -> also covers 22
    #   L=21 -> dist=20 -> chord (0,20) -> covers 21
    #   L=23 -> dist=18 -> chord (0,18) -> covers 23 and 23? No: 41-18=23. Yes!
    #   Wait, 41-18=23. So chord (0,18) covers 19 and 23.
    #   L=19 -> dist=18 -> chord (0,18) covers 19 and 23.
    #   L=24 -> dist=17 -> chord (0,17) covers 18 and 24.
    #   L=18 -> dist=17 -> chord (0,17) covers 18 and 24.
    #   ...
    # Let me list all chords (0, k) for k=2..20:
    #   k=2:  covers 3, 39
    #   k=3:  covers 4, 38
    #   k=4:  covers 5, 37
    #   k=5:  covers 6, 36
    #   k=6:  covers 7, 35
    #   k=7:  covers 8, 34
    #   k=8:  covers 9, 33
    #   k=9:  covers 10, 32
    #   k=10: covers 11, 31
    #   k=11: covers 12, 30
    #   k=12: covers 13, 29
    #   k=13: covers 14, 28
    #   k=14: covers 15, 27
    #   k=15: covers 16, 26
    #   k=16: covers 17, 25
    #   k=17: covers 18, 24
    #   k=18: covers 19, 23
    #   k=19: covers 20, 22
    #   k=20: covers 21, 21
    # That's 19 chords covering all lengths 3..39!
    # Total edges = 40 + 19 = 59. Same as the chorded Hamiltonian cycle.

    # To do BETTER than 59, we need chords that cover lengths NOT just via the
    # simple "arc+chord" cycle. E.g., a cycle using TWO chords.
    # Consider chord (0,2) and chord (5,7). A cycle could be:
    #   0-1-2 (chord back? no) ... 
    # Actually, let me think about a different construction entirely.

    # ALTERNATIVE: Use a "ladder" or "grid" structure.
    # Consider a 4x10 grid. 40 vertices. 
    # Grid has 4*10 vertices, edges: horizontal 4*(10-1)=36, vertical (4-1)*10=30. Total 66. Too many.
    # Plus we need to make it pancyclic.

    # Let me try: a path with selected chords, but optimize which chords.
    # Key idea: instead of all chords from vertex 0, use chords that "overlap"
    # in their covered lengths, reducing the total number of chords needed.

    # Actually, the 59-edge construction is likely near-optimal for this approach.
    # Let me try to find a sparser construction.

    # NEW IDEA: Use a smaller "core" structure.
    # A pancyclic graph on n vertices needs at least n edges (Hamiltonian cycle).
    # For n=40, minimum possible is 40 (just the Hamiltonian cycle), but that only
    # gives one cycle (length 40). We need 37 more cycle lengths.
    # Each added edge can potentially create multiple new cycle lengths.

    # Let's try: Hamiltonian cycle + 12 carefully chosen chords.
    # If each chord on average covers ~3 lengths (via combinations), 12 chords might suffice.
    # But this is hard to guarantee. Let me just implement the 59-edge version
    # and then try to delete edges greedily.

    # Build the 59-edge graph: Hamiltonian cycle + chords (0,k) for k=2..20
    edges = set()
    for i in range(N):
        edges.add(norm((i, (i + 1) % N)))
    for k in range(2, 21):
        edges.add(norm((0, k)))

    # Now extract cycles by construction:
    # For each L in 3..39, find the chord k such that dist[0][k]=L-1 or 41-L=L... 
    # Actually, for L in 3..20: chord (0, L-1), cycle = [0, 1, 2, ..., L-1, 0]
    #   Wait, that's a path 0->1->...->L-1 plus chord (L-1, 0). Length = L.
    #   But only if L-1 <= 20 (i.e., L <= 21). For L=3: path 0-1-2, chord (2,0). Length 3. ✓
    # For L in 21..39: chord (0, 41-L), cycle = [0, 39, 38, ..., 41-L, 0]
    #   Path 0->39->38->...->(41-L) has length (0 - (41-L)) mod 40 = L-1 edges? 
    #   Let me count: from 0 going backwards: 0, 39, 38, ..., (41-L). 
    #   Number of vertices = 0 down to (41-L) = 40 - (41-L) = L-1 vertices.
    #   Plus the chord (41-L, 0). Total vertices in cycle = L-1 + 1 = L. ✓
    #   Wait, vertices: 0, 39, 38, ..., (41-L). That's (0 - (41-L) + 1) = L-1+1 = L vertices? 
    #   From 41-L to 0 going up: 41-L, 42-L, ..., 39, 0. Count = 0 - (41-L) = L-1. Plus 0 = L vertices.
    #   Hmm, let me just be explicit.

    cycles = {}

    # Length 40: the Hamiltonian cycle
    cycles[40] = list(range(N))

    for L in range(3, 40):
        if L <= 21:
            # Use chord (0, L-1), forward arc
            k = L - 1
            # Cycle: 0, 1, 2, ..., k, 0  (k+1 = L vertices)
            cycle = list(range(0, k + 1))
        else:
            # Use chord (0, 41-L), backward arc
            k = 41 - L
            # Cycle: 0, 39, 38, ..., k, 0
            # Vertices from 0 going backwards to k: 0, 39, 38, ..., k
            # That's (0 - k) mod 40 = 40 - k vertices, plus we close with chord.
            # Actually: 0, 39, 38, ..., k. Count = 40 - k.
            # k = 41-L, so count = 40 - (41-L) = L-1. But we need L vertices!
            # The cycle is: 0, 39, 38, ..., k, then chord (k,0) back to 0.
            # Vertices: 0, 39, 38, ..., k. That's 40 - k + 1 = 41 - k = L vertices. ✓
            cycle = [0] + list(range(39, k - 1, -1))
        cycles[L] = cycle

    # Verify all cycles are valid
    adj = build_adjacency(edges)
    valid = True
    for L in range(3, N + 1):
        c = cycles[L]
        if len(c) != L:
            valid = False
            break
        if len(set(c)) != L:
            valid = False
            break
        for i in range(L):
            u = c[i]
            v = c[(i + 1) % L]
            if v not in adj[u]:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: shouldn't happen, but return the graph with best-effort cycles
        return {'edges': sorted(edges), 'cycles': [cycles[L] for L in range(3, N + 1)]}

    # Now try to delete edges to reduce count
    # The Hamiltonian cycle edges are critical (needed for length 40 and many others)
    # The chords are candidates for deletion
    # Try deleting each chord and re-checking pancyclicity

    def is_pancyclic(edge_set):
        """Check if graph with given edges is pancyclic (has cycles of all lengths 3..40)."""
        if len(edge_set) < N:
            return False  # Can't have Hamiltonian cycle
        adj = build_adjacency(edge_set)
        # Check length 40: need a Hamiltonian cycle
        # For efficiency, just check if the original Hamiltonian cycle still exists
        # (we only delete chords, so the Hamiltonian cycle edges remain)
        # Actually, we might delete Hamiltonian cycle edges too. Let's do a proper check.
        # But that's expensive. Let's restrict deletion to chords only.
        for L in range(3, N + 1):
            # Try to find a cycle of length L
            # Use the precomputed cycle if it's still valid
            c = cycles[L]
            ok = True
            for i in range(L):
                u = c[i]
                v = c[(i + 1) % L]
                if v not in adj[u]:
                    ok = False
                    break
            if not ok:
                # Try to find any cycle of length L (expensive, only for small L)
                # For now, if the precomputed cycle is broken, consider it not pancyclic
                return False
        return True

    # Try deleting chords (non-Hamiltonian-cycle edges)
    h_cycle_edges = set(norm((i, (i + 1) % N)) for i in range(N))
    chord_edges = [e for e in edges if e not in h_cycle_edges]
    random.seed(123)
    random.shuffle(chord_edges)

    for chord in chord_edges:
        if time.time() - start_time > deadline - 20:
            break
        if chord not in edges:
            continue
        test_edges = set(edges)
        test_edges.discard(chord)
        if is_pancyclic(test_edges):
            edges = test_edges

    # Final verification
    adj = build_adjacency(edges)
    for L in range(3, N + 1):
        c = cycles[L]
        for i in range(L):
            u = c[i]
            v = c[(i + 1) % L]
            if v not in adj[u]:
                # Rebuild: if we deleted a chord that broke a cycle, we need to find a new one
                # For robustness, do a DFS search for this cycle length
                found = False
                # Simple DFS for cycle of length L
                def dfs_cycle(start, target_len):
                    path = [start]
                    visited = {start}
                    def _dfs(p, vis):
                        if len(p) == target_len:
                            if p[0] in adj[p[-1]]:
                                return list(p)
                            return None
                        for nxt in adj[p[-1]]:
                            if nxt not in vis and nxt != p[0]:
                                p.append(nxt)
                                vis.add(nxt)
                                r = _dfs(p, vis)
                                if r:
                                    return r
                                p.pop()
                                vis.remove(nxt)
                        return None
                    return _dfs(path, visited)
                
                for s in range(N):
                    c_found = dfs_cycle(s, L)
                    if c_found:
                        cycles[L] = c_found
                        found = True
                        break
                if not found:
                    # Can't fix, return what we have (may be invalid, but best effort)
                    break

    edge_list = sorted(edges)
    cycles_out = []
    for L in range(3, N + 1):
        cycles_out.append(cycles[L])

    return {'edges': edge_list, 'cycles': cycles_out}

# EVOLVE-BLOCK-END

# Call the construct function
construct()
