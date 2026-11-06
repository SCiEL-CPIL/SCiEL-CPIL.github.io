import time

N = 40

def construct():
    t_start = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill
    
    # Base construction: cycle + chords from vertex 0
    # This gives 59 edges and covers all cycle lengths 3..40
    edges_base = []
    for i in range(N):
        edges_base.append((i, (i + 1) % N))
    for k in range(2, 21):
        edges_base.append((0, k))
    
    edges_set = set()
    for e in edges_base:
        edges_set.add((min(e), max(e)))
    
    cycles_base = []
    for L in range(3, 22):
        cycles_base.append(list(range(L)))
    for L in range(22, 40):
        k = N - L + 1
        cycles_base.append([0] + list(range(k, N)))
    cycles_base.append(list(range(N)))
    
    best_edges = list(edges_set)
    best_cycles = cycles_base
    best_score = len(best_edges) - N
    
    # Try to improve by finding alternative constructions
    # Strategy: use a different set of chords that might allow fewer total edges
    # by combining with other edge types more efficiently
    
    # Try: cycle + chords from multiple hub vertices
    # This is a different structural family
    
    import random
    rng = random.Random(42)
    
    # Try random chord sets from vertex 0 (different subsets)
    # We need chords such that for every L in 3..39, there exists a cycle of length L
    # Using vertex 0 as hub: chord (0,k) gives cycle [0,1,...,k] of length k+1
    # and [0,k,k+1,...,N-1] of length N-k+1
    # So we need for each L: either L-1 in chord_set (giving length L) 
    # or N-L+1 in chord_set (giving length L via the other direction)
    # chord_set must contain values in 2..N-2 (excluding 1 and N-1 which are cycle edges)
    
    # Current: chords {2,3,...,20}
    # For L=3: need chord 2 (gives [0,1,2]) or chord N-3+1=38 (gives [0,38,39])
    # For L=4: need chord 3 or chord 37
    # ...
    # For L=21: need chord 20 or chord 20
    # For L=22: need chord 21 or chord 19
    # For L=39: need chord 38 or chord 2
    
    # So we need a set S ⊆ {2,...,38} such that for every L in 3..39:
    # (L-1) ∈ S or (N-L+1) ∈ S
    # i.e., for every j in 2..38: j ∈ S or (N-j+1) ∈ S... wait let me re-derive
    
    # Cycle length L via chord (0,k):
    # Path 0-1-2-...-k-0: length k+1 (vertices 0,1,...,k)
    # Path 0-k-(k+1)-...-(N-1)-0: length N-k+1 (vertices 0,k,k+1,...,N-1)
    # So chord k gives lengths {k+1, N-k+1}
    # We need for each L in 3..39: ∃k such that L ∈ {k+1, N-k+1}
    # L = k+1 → k = L-1
    # L = N-k+1 → k = N-L+1
    # So for each L: (L-1) ∈ S or (N-L+1) ∈ S
    # Let j = L-1, then j ranges 2..38
    # Condition: j ∈ S or (N-(j+1)+1) ∈ S → j ∈ S or (N-j) ∈ S
    # So for each j in 2..38: j ∈ S or (N-j) ∈ S
    
    # This is a covering problem. Minimum |S|?
    # Pairs: {2,38}, {3,37}, {4,36}, ..., {19,21}, {20,20}
    # 19 pairs, we need at least one from each pair
    # Minimum |S| = 19 (pick one from each pair)
    # Current solution picks {2,3,...,20} which is 19 elements ✓
    
    # So 19 chords is minimum for this hub-based approach.
    # Total edges = 40 (cycle) + 19 (chords) = 59.
    
    # Can we do better with a non-hub approach?
    # A pancyclic graph on n vertices needs at least n edges (it's a cycle).
    # With n edges, it's exactly a cycle, which only has one cycle of length n.
    # So we need more edges.
    
    # Lower bound: for a graph to have cycles of all lengths 3..n,
    # it needs to be pancyclic. Known results suggest n + O(log n) edges suffice.
    # For n=40, 59 = 40+19. Can we get 40+18=58?
    
    # With 18 chords from vertex 0: we cover 18 pairs out of 19, missing one length.
    # So hub-only can't do 18 chords.
    
    # What if we add edges not incident to 0?
    # E.g., add edge (a,b) where a,b ≠ 0. This might create additional cycles.
    # But we'd still need to cover all 19 pairs, so we'd need 19 "effective" chords
    # plus possibly extra edges, making it worse.
    
    # Alternative: use a different base structure.
    # E.g., two overlapping cycles, or a theta graph structure.
    
    # Try: complete graph K_5 on vertices 0-4, plus path 4-5-6-...-39
    # K_5 has 10 edges, path has 35 edges, total 45 edges.
    # But does it have cycles of all lengths? Probably not.
    
    # Let's try a computational search for a 58-edge solution.
    # This is complex, so let's try a heuristic.
    
    # Approach: start with 59-edge solution, try to remove one edge and 
    # see if we can add a different edge to maintain pancyclicity.
    
    # Actually, let's try a completely different construction:
    # Use a "ladder" or "grid" like structure.
    
    # For now, return the known 59-edge solution.
    if time.time() > deadline:
        return {'edges': best_edges, 'cycles': best_cycles}
    
    # Try random perturbations to see if we can find 58 edges
    # This is unlikely but let's try a few
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Start with base, remove a random chord, add a random non-incident edge
        chords = list(range(2, 21))
        if not chords:
            break
        
        remove_idx = rng.randint(0, len(chords) - 1)
        removed_chord = chords.pop(remove_idx)
        
        # New edges: cycle + remaining chords + one new edge
        new_edges = set()
        for i in range(N):
            new_edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        for k in chords:
            new_edges.add((0, k))
        
        # Add a random edge not in the set
        candidates = []
        for i in range(1, N):
            for j in range(i+1, N):
                if (i,j) not in new_edges and i != 0 and j != 0:
                    candidates.append((i,j))
        
        if not candidates:
            continue
        
        new_edge = rng.choice(candidates)
        new_edges.add(new_edge)
        
        # Check if this graph is pancyclic (has cycles of all lengths 3..40)
        # This is expensive, so do a quick check
        # Build adjacency
        adj = [set() for _ in range(N)]
        for (u,v) in new_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Quick check: does it have a cycle of each length?
        # Use DFS for small cycles, but this is expensive for all lengths
        # Skip for now - just return the 59-edge solution
        
        # If we had a fast pancyclicity checker, we'd use it here
    
    return {'edges': best_edges, 'cycles': best_cycles}
