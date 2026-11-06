N = 40

def construct():
    # Base cycle C40: edges (i, i+1) for i=0..38, and (39, 0)
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    # Chords (0, d) for d = 2..20
    chords = [(0, d) for d in range(2, 21)]
    edges = base_edges + chords
    
    # Deduplicate and sort
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    edges = sorted(edge_set)
    
    # Build cycle certificates for each length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            # Cycle: 0 -> 1 -> 2 -> ... -> k-1 -> 0
            # Uses path edges (0,1), (1,2), ..., (k-2,k-1) and chord (0, k-1)
            # For k=21, chord is (0,20) which exists
            cycle = list(range(k))
        else:
            # k > 21: use chord (0, d) where d = 41 - k (so d is in 2..19)
            # Cycle: 0 -> d -> d+1 -> ... -> 39 -> 0
            # Uses chord (0,d), path edges (d,d+1),...,(38,39), and edge (39,0)
            d = 41 - k
            cycle = [0] + list(range(d, N))
        cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
