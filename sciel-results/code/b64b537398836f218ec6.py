"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Build a sparse pancyclic graph on 40 vertices.
    
    Structure:
    - Hamiltonian rim: cycle 0-1-2-...-39-0 (40 edges)
    - Chords from vertex 0 to vertices 2 through 20 (19 chords)
    Total: 59 edges, score = 59 - 40 = 19
    
    Cycle certificates:
    - Length k (3 <= k <= 21): 0, 1, 2, ..., k-1, 0 (uses chord (0, k-1) for k>=3, 
      but actually uses rim edges and chord (0,k-1))
      Wait, let me reconsider. For k=3: 0,1,2,0 uses edges (0,1),(1,2),(2,0) - chord (0,2)
      For k=4: 0,1,2,3,0 uses edges (0,1),(1,2),(2,3),(3,0) - chord (0,3)
      ...
      For k=21: 0,1,2,...,20,0 uses chord (0,20)
      
    - Length k (22 <= k <= 39): Need cycles of these lengths.
      Use: 0, 39, 38, ..., (40-k+1), 0
      For k=22: 0, 39, 38, ..., 19, 0 (22 vertices: 0,39,38,...,19)
      Edges: (0,39) rim, (39,38) rim, ..., (20,19) rim, (19,0) chord
      Chord needed: (0,19) - but we only have chords up to (0,20). (0,19) is in range!
      
      Let me verify: for k=22, start = N-k+1 = 40-22+1 = 19
      Cycle: 0, 39, 38, ..., 19, 0
      Vertices: 0, 39, 38, 37, ..., 19
      Count: 1 + (39-19+1) = 1 + 21 = 22. Correct.
      Edges: (0,39) rim, (39,38) rim, ..., (20,19) rim, (19,0) chord.
      (0,19) is a chord we have (since 19 is in 2..20).
      
      For k=39: start = 40-39+1 = 2
      Cycle: 0, 39, 38, ..., 2, 0
      Vertices: 0, 39, 38, ..., 2
      Count: 1 + (39-2+1) = 1 + 38 = 39. Correct.
      Edges: (0,39) rim, (39,38) rim, ..., (3,2) rim, (2,0) chord.
      (0,2) is a chord we have.
      
    - Length 40: Hamiltonian cycle 0, 1, 2, ..., 39, 0
    """
    deadline = time.time() + 80  # 80s deadline, 10s margin
    
    edges = set()
    
    # Hamiltonian rim
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    # Chords from 0 to vertices 2 through 20
    for j in range(2, 21):
        edges.add((0, j))
    
    edge_list = sorted(edges)
    
    # Build cycle certificates for every length 3..40
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            # Cycle: 0, 1, 2, ..., k-1, 0
            cycle = list(range(k))
        elif k < N:
            # Cycle: 0, 39, 38, ..., (N-k+1), 0
            start = N - k + 1
            cycle = [0] + list(range(N - 1, start - 1, -1))
        else:
            # Hamiltonian cycle: 0, 1, 2, ..., 39, 0
            cycle = list(range(N))
        cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
