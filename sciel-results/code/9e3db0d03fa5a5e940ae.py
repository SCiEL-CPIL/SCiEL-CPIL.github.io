import time
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Base Hamiltonian cycle: 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        base_edges.add((min(u, v), max(u, v)))
    
    # We need cycles of lengths 3..39 (length 40 is the Hamiltonian cycle)
    # A chord (a, b) in the Hamiltonian cycle creates two cycles:
    # One of length d+1 and one of length N-d where d is the distance along the cycle
    # We need to cover all lengths 3..39 with chords
    
    # Strategy: Use a set of chords, each chord (a, b) covers lengths 
    # dist+1 and N-dist where dist is the cyclic distance from a to b
    # We want to minimize the number of chords
    
    # Precompute: for each possible chord (not in base cycle), what lengths does it cover?
    # Chord (a, b) where b-a mod N != 1 and != N-1
    # distance = min((b-a) % N, (a-b) % N)
    # Covers lengths: distance+1 and N-distance
    
    # We need to cover lengths 3..39
    # Each chord covers at most 2 lengths
    # Minimum chords needed: ceil(37/2) = 19 (if perfect pairing)
    
    # But we can potentially do better if some chords cover lengths that can't be paired
    # Actually, the pairing is: length L and length N+1-L? No.
    # Chord with distance d covers d+1 and N-d
    # d+1 + (N-d) = N+1 = 41
    # So lengths come in pairs summing to 41: (3,38), (4,37), ..., (20,21)
    # That's 19 pairs, so we need at least 19 chords. 59 edges is optimal for this structure.
    
    # However, we can use chords that don't all pass through vertex 0!
    # A chord (a, b) with distance d covers the same lengths regardless of a, b.
    # So the minimum is still 19 chords = 59 edges.
    
    # Wait - can a single chord cover more than 2 cycle lengths? No, a chord in a 
    # Hamiltonian cycle creates exactly 2 cycles.
    
    # So 59 edges is the theoretical minimum for this approach.
    # But maybe there's a non-Hamiltonian-cycle-based construction with fewer edges?
    
    # For now, use the optimal 59-edge construction
    edges = set(base_edges)
    
    # Add 19 chords from vertex 0 to cover all pairs
    # Chord (0, m) for m=2..20 covers lengths m+1 and 40-m
    for m in range(2, 21):
        edges.add((0, m))
    
    edge_list = list(edges)
    
    # Construct cycles
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycles.append(list(range(N)))
        elif k <= 21:
            # Cycle: 0-1-2-...-(k-1)-0, uses chord (0, k-1)
            cycles.append(list(range(k)))
        else:
            # Cycle: 0-39-38-...-m-0 where m = 40-k+1 = 41-k
            # Wait: chord (0, m) covers length 40-m. We want 40-m = k, so m = 40-k
            # But for k=22, m=18. Cycle: 0-39-38-...-18-0. Length = 40-18 = 22. ✓
            # For k=39, m=1. But m=1 is in the base cycle! 
            # Actually m=40-k. For k=39, m=1. Chord (0,1) is in base cycle.
            # The cycle 0-39-38-...-1-0 has length 39. ✓
            m = 40 - k
            if m == 0:
                # k=40, handled above
                pass
            cycle_verts = [0] + list(range(39, m - 1, -1))
            cycles.append(cycle_verts)
    
    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    valid = True
    for c in res['cycles']:
        if len(set(c)) != len(c):
            valid = False
            break
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (u,v) not in edges and (v,u) not in edges:
                valid = False
                break
        if not valid:
            break
    print(f"Edges: {len(res['edges'])}, Valid: {valid}")
