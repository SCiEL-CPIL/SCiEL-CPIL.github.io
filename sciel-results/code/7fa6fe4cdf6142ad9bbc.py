"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40
START_TIME = time.time()
DEADLINE = 90  # seconds, keep margin

def elapsed():
    return time.time() - START_TIME

def check_validity(edges, cycles):
    """Check if all cycles are valid and all required edges exist."""
    edge_set = set()
    for e in edges:
        a, b = e
        if a == b:
            return False
        edge_set.add((min(a, b), max(a, b)))
    
    for k, cycle in enumerate(cycles, start=3):
        if len(cycle) != k:
            return False
        # Check distinct vertices
        if len(set(cycle)) != k:
            return False
        # Check all vertices in range
        for v in cycle:
            if v < 0 or v >= N:
                return False
        # Check consecutive edges exist
        for i in range(k):
            u = cycle[i]
            v = cycle[(i + 1) % k]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
    return True

def try_prune_edges(edges, cycles):
    """Try to remove edges one by one while maintaining validity."""
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
    
    current_edges = list(edge_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(current_edges)):
            if elapsed() > DEADLINE - 5:
                break
            trial_edges = [e for j, e in enumerate(current_edges) if j != i]
            trial_set = set((min(a,b), max(a,b)) for a,b in trial_edges)
            # Quick check: verify each cycle
            valid = True
            for k, cycle in enumerate(cycles, start=3):
                for idx in range(k):
                    u = cycle[idx]
                    v = cycle[(idx + 1) % k]
                    if (min(u,v), max(u,v)) not in trial_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_edges = trial_edges
                changed = True
                break
        if elapsed() > DEADLINE - 5:
            break
    
    return [(a, b) for a, b in current_edges]

# EVOLVE-BLOCK-START
def construct():
    if elapsed() > DEADLINE - 5:
        # Fallback: return a valid but non-optimal solution
        edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        cycles = [list(range(k)) for k in range(3, N+1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Structural approach: base cycle + strategic chords
    # Base: Hamiltonian cycle 0-1-2-...-39-0
    base_edges = [(i, (i+1) % N) for i in range(N)]
    
    # We need cycles of lengths 3..39
    # A chord (i, j) creates a cycle of length min(|i-j|, N-|i-j|) + 1... 
    # Actually, chord (i,j) where d = min(|i-j|, N-|i-j|) creates cycles of length d and N-d
    # Wait: if we have the base cycle and add chord (i,j), we get two cycles:
    # one going i->i+1->...->j (length |i-j|+1 if we count edges, but vertices = |i-j|+1)
    # Hmm, let me think more carefully.
    
    # In a cycle 0-1-2-...-39-0, adding chord (i,j) splits it into two cycles:
    # Cycle 1: i, i+1, ..., j (going forward) - has |i-j| edges on the path + 1 chord = |i-j|+1 edges, so |i-j|+1 vertices
    # Cycle 2: j, j+1, ..., i (going forward) - has (N-|i-j|) edges on path + 1 chord = N-|i-j|+1 edges, so N-|i-j|+1 vertices
    # Wait no. The path from i to j going forward has (j-i) mod N edges. The cycle formed has (j-i) mod N + 1 vertices.
    
    # Let's use a different approach: greedily add chords to cover all cycle lengths 3..39
    # Start with just the base cycle (gives us length 40)
    # We need to find chords that create cycles of each length 3..39
    
    # Key insight: a chord between vertices at distance d (along the cycle) creates a cycle of length d+1 
    # (the shorter arc) and N-d+1... no wait.
    
    # In cycle v0-v1-...-v39-v0, chord (vi, vj):
    # Path 1: vi -> vi+1 -> ... -> vj (mod N), length = (j-i) mod N edges, so (j-i) mod N + 1 vertices
    # Path 2: vj -> vj+1 -> ... -> vi (mod N), length = (i-j) mod N edges, so (i-j) mod N + 1 vertices
    # So we get cycles of lengths (j-i) mod N + 1 and (i-j) mod N + 1
    # Note: ((j-i) mod N) + ((i-j) mod N) = N (when i != j)
    # So the two cycle lengths sum to N+1 = 41
    
    # So each chord gives us two cycle lengths that sum to 41.
    # We need lengths 3..39 (37 lengths). Each chord covers at most 2 new lengths.
    # So we need at least ceil(37/2) = 19 chords.
    # Total edges: 40 + 19 = 59, score = 19.
    
    # But we can be smarter: some cycles might be formed by combinations of chords,
    # or we might find that certain chords cover multiple needed lengths.
    
    # Let's try: for each needed length L (3..39), we need a chord that creates a cycle of length L.
    # A chord creating cycle of length L means the distance along the cycle is L-1.
    # So chord (i, i+L-1) mod N creates a cycle of length L.
    
    # We want to minimize the number of chords. Each chord (i,j) with dist = min((j-i)%N, (i-j)%N)
    # creates cycles of length dist+1 and N-dist+1.
    
    # Let's greedily select chords to cover all lengths 3..39.
    needed = set(range(3, 40))  # lengths 3..39
    chords = []
    
    # For each possible chord distance d (1..19), the chord creates cycles of length d+1 and 41-d
    # Distance d means the shorter arc has d edges, so cycle has d+1 vertices
    # The other arc has N-d edges, so cycle has N-d+1 = 41-d vertices
    
    # We want to pick chords (specific pairs) to cover all needed lengths.
    # A chord at distance d covers lengths {d+1, 41-d}.
    
    # Let's enumerate all possible chords and do greedy set cover.
    # But there are many chords. Let's be smart: for each distance d, we just need one chord.
    # Actually, any chord at distance d will create cycles of length d+1 and 41-d regardless of position.
    
    # So the problem reduces to: choose a set of distances D such that for each L in 3..39,
    # either L-1 in D or 41-L-1 = 40-L in D.
    # i.e., for each L in 3..39, either (L-1) in D or (40-L) in D.
    
    # Let's map: for L, we need distance L-1 or distance 40-L.
    # L=3: need dist 2 or dist 37
    # L=4: need dist 3 or dist 36
    # ...
    # L=20: need dist 19 or dist 20 -> but dist 20 is same as dist 20 (since min(d, 40-d), dist 20 means the other is also 20)
    # Wait, for N=40, distance d and 40-d are the same chord (just opposite direction).
    # So effective distances are 1..20.
    # For L, we need dist = min(L-1, 40-(L-1)) = min(L-1, 41-L)
    
    # Let me recompute: chord at position distance d (where d = (j-i) % N, we take min(d, N-d) as the "short distance")
    # creates cycles of length d+1 and N-d+1 where d is the short distance.
    
    # For needed length L, we need a chord with short distance d such that d+1 = L or N-d+1 = L
    # i.e., d = L-1 or d = N-L+1 = 41-L
    # Since d is the short distance (1..20), we need min(L-1, 41-L) to be in our chosen distances.
    
    # So for each L in 3..39, define required_dist(L) = min(L-1, 41-L)
    # L=3: min(2,38)=2
    # L=4: min(3,37)=3
    # L=5: min(4,36)=4
    # ...
    # L=20: min(19,21)=19
    # L=21: min(20,20)=20
    # L=22: min(21,19)=19
    # L=23: min(22,18)=18
    # ...
    # L=39: min(38,2)=2
    
    # So the required distances are:
    # L=3,39 -> 2
    # L=4,38 -> 3
    # L=5,37 -> 4
    # ...
    # L=20 -> 19
    # L=21 -> 20
    # L=22 -> 19
    # L=23 -> 18
    # ...
    # L=39 -> 2
    
    # Unique required distances: 2,3,4,...,19,20
    # That's 19 distinct distances!
    # So we need chords at each of these 19 distances.
    # Total edges: 40 + 19 = 59.
    
    # But wait - can a single chord cover multiple required distances? No, each chord has exactly one short distance.
    # So minimum is 19 chords, giving 59 edges, score 19.
    
    # Can we do better? What if we don't use a simple cycle as base?
    # Or what if some cycles can be formed without dedicated chords?
    
    # Actually, the base cycle itself gives length 40. We need 3..39.
    # With 19 chords, we cover all of 3..39. Total 59 edges.
    
    # Let's construct this: base cycle + one chord for each distance 2..20.
    
    edges_set = set()
    for e in base_edges:
        a, b = e
        edges_set.add((min(a,b), max(a,b)))
    
    # Add chords for distances 2..20
    # For distance d, add chord (0, d) -- this creates cycles of length d+1 and 41-d
    for d in range(2, 21):
        a, b = 0, d
        edges_set.add((min(a,b), max(a,b)))
    
    edges = list(edges_set)
    
    # Now construct cycles for each length 3..39
    # For length L, we need a cycle of exactly L vertices.
    # If L-1 <= 20 (i.e., L <= 21): use chord (0, L-1) and the path 0,1,...,L-1
    #   Cycle: 0, 1, 2, ..., L-1, 0 (using chord (0,L-1))
    #   Wait, that's L vertices: 0,1,...,L-1 is L vertices.
    #   Edges: (0,1),(1,2),...,(L-2,L-1),(L-1,0) - all in our edge set?
    #   (L-1,0) is the chord we added. Yes!
    
    # If L-1 > 20 (i.e., L >= 22): use chord (0, 41-L) and the longer path
    #   41-L < 20 for L > 21. Let d = 41-L.
    #   Chord (0, d) creates a cycle of length 41-d = L.
    #   The cycle goes: 0, 39, 38, ..., d, 0 (the long way around)
    #   Vertices: 0, 39, 38, ..., d. Count: 40-d+1 = 41-d = L. Yes!
    #   Edges: (0,39),(39,38),...,(d+1,d),(d,0) - all in base cycle except (d,0) which is the chord.
    
    cycles = []
    for L in range(3, 40):
        if L <= 21:
            # Short arc: 0, 1, 2, ..., L-1
            cycle = list(range(L))
        else:
            # Long arc: 0, 39, 38, ..., 41-L
            d = 41 - L
            cycle = [0] + list(range(39, d-1, -1))
            # Verify: 0, 39, 38, ..., d. That's 1 + (39-d+1) = 40-d = 41-L = L vertices.
        cycles.append(cycle)
    
    # Add the Hamiltonian cycle (length 40)
    cycles.append(list(range(N)))
    
    # Verify validity
    if not check_validity(edges, cycles):
        # Fallback to seed
        edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
        cycles = [list(range(k)) for k in range(3, N+1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Try to prune
    if elapsed() < DEADLINE - 10:
        edges = try_prune_edges(edges, cycles)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
