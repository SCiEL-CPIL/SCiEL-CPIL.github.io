import time

N = 40

def construct():
    """
    Construct a sparse pancyclic graph on 40 vertices with 59 edges.
    
    Design: A Hamiltonian cycle on vertices 0..39, plus 19 chords from vertex 0
    to vertices 2..20. This gives 40 + 19 = 59 edges.
    
    The chords enable cycles of all lengths 3..40:
    - For length L (3 <= L <= 21): use chord (0, L-1) and path 0-1-...-(L-1)
    - For length L (22 <= L <= 39): use chord (0, 41-L) and path 0-(41-L)-...-39
    - For length 40: the Hamiltonian cycle itself
    """
    edges = set()
    
    # Hamiltonian cycle: (0,1), (1,2), ..., (38,39), (39,0)
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # Chords from vertex 0 to vertices 2, 3, ..., 20
    for k in range(2, N // 2 + 1):
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # Construct cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle_verts = list(range(N))
        elif L <= N // 2 + 1:
            # Use chord (0, L-1): path 0 -> 1 -> ... -> (L-1) -> 0
            # Length = (L-1) + 1 = L
            cycle_verts = list(range(L))
        else:
            # Use chord (0, N+1-L): path 0 -> (N+1-L) -> (N+2-L) -> ... -> (N-1) -> 0
            # Path length = N-1 - (N+1-L) + 1 = L-2, plus chord = L-1 edges... 
            # Wait, let me recount.
            # Vertices: 0, (N+1-L), (N+2-L), ..., (N-1)
            # Number of vertices = 1 + (N-1 - (N+1-L) + 1) = 1 + (L-2) = L-1
            # That's not L vertices. Let me reconsider.
            # 
            # Actually: vertices are 0, k, k+1, ..., N-1 where k = N+1-L
            # Count: 1 + (N-1 - k + 1) = 1 + (N - k) = 1 + (N - (N+1-L)) = 1 + (L-1) = L
            # Edges: (0,k), (k,k+1), ..., (N-2,N-1), (N-1,0)
            # That's 1 + (N-1-k) + 1 = N-k+1 = N-(N+1-L)+1 = L-1+1 = L edges. Correct.
            k = N + 1 - L
            cycle_verts = [0] + list(range(k, N))
        
        cycles.append(cycle_verts)
    
    return {'edges': edge_list, 'cycles': cycles}
