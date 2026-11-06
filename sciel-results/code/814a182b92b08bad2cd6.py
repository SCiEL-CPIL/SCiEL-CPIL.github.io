import time

N = 40

def construct():
    # Deadline guard: 90s limit, use 80s for safety
    deadline = time.time() + 80
    
    # 1. Base Cycle Edges
    # Vertices 0..39
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # 2. Add Chords
    # To cover cycle lengths 3..39 using chords from hub 0 on a Hamiltonian cycle:
    # A chord (0, k) creates cycles of length k+1 and N-k+1.
    # We need to cover all L in 3..39.
    # L=3 -> k=2 (or 38->2)
    # L=4 -> k=3 (or 37->3)
    # ...
    # L=21 -> k=20 (or 21->19? No, 41-21=20)
    # L=22 -> k=21 (equiv 19) or k=19.
    # Analysis shows chords k=2..20 are required and sufficient.
    
    # Check time occasionally, though this loop is trivial
    if time.time() > deadline:
        # Fallback: return base cycle (invalid for pancyclic, but avoids crash)
        # Actually, we should return the best valid graph found so far.
        # Since we are just adding edges, we can return what we have.
        # But we must have a valid graph. 
        # The loop is so fast it will never timeout.
        pass

    chords_k = list(range(2, N // 2 + 1)) # 2..20
    
    for k in chords_k:
        e = (0, k)
        edges.add(e)
        
    edge_list = [tuple(sorted(e)) for e in edges]
    
    # 3. Construct Cycles
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            # Try to use chord (0, L-1)
            k1 = L - 1
            k2 = N + 1 - L
            
            # We added chords for k in 2..20.
            # If k1 is in 2..20, use it.
            if 2 <= k1 <= N // 2:
                # Cycle: 0 -> 1 -> ... -> k1 -> 0
                # Vertices: 0, 1, ..., k1
                # Length: k1 - 0 + 1 (path) + 1 (chord) = k1 + 1 = L.
                cycle_verts = list(range(k1 + 1))
            else:
                # k1 > 20. Then k2 < 20.
                # Also k2 >= 2 because L <= N-1 = 39 => k2 = 41-L >= 2.
                # So k2 is in 2..19, which is in our chord set.
                # Use chord (0, k2).
                # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                # Vertices: 0, k2, k2+1, ..., N-1
                # Path length: N-1 - k2 + 1 = N - k2.
                # Plus chord: N - k2 + 1 = (40 - k2) + 1 = 41 - k2 = L.
                cycle_verts = [0] + list(range(k2, N))
                
        cycles.append(cycle_verts)
        
    return {'edges': edge_list, 'cycles': cycles}
