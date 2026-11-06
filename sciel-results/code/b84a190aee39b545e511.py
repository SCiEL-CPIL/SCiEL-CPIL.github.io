"""Constructs a sparse pancyclic graph on 40 vertices."""

N = 40

def construct():
    # 1. Base cycle edges
    edges = [(i, (i + 1) % N) for i in range(N)]
    
    # 2. Chords to ensure pancyclic property
    # We add chords (0, i) for i in 2..20.
    # This covers circular distances 2..20 and 20..38.
    # Distance d corresponds to cycle length d+1.
    # Distances covered: 2..38 -> Cycle lengths 3..39.
    # Length 40 is the base cycle.
    for i in range(2, 21):
        edges.append((0, i))
    
    # Remove duplicates if any (though (0,i) for i>=2 are distinct from cycle edges (i, i+1) and (39,0))
    # Cycle edges are (0,1), (1,2), ..., (38,39), (39,0).
    # Chords are (0,2), ..., (0,20).
    # No overlap.
    
    # 3. Construct cycles for lengths 3 to 40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Base cycle
            cycles.append(list(range(N)))
        else:
            d = k - 1
            if d <= 20:
                # Chord (0, d) closes the path 0->1->...->d
                cycles.append(list(range(d + 1)))
            else:
                # Chord (0, N-d) closes the path 0->N-1->...->N-d
                # The vertices are 0, N-1, N-2, ..., N-d
                cycle_vertices = [0] + list(range(N - 1, N - d - 1, -1))
                cycles.append(cycle_vertices)
                
    return {'edges': edges, 'cycles': cycles}
