"""
Constructs a sparse pancyclic graph on 40 vertices.
Target: Minimize (edge_count - 40).

Design:
- Base Hamiltonian cycle: 0-1-2-...-39-0 (40 edges).
- Add chords from vertex 0 to vertices 2, 3, ..., 20 (19 chords).
  Total edges: 40 + 19 = 59. Value: 59 - 40 = 19.

Cycle construction:
- For L in 3..21: cycle is 0, 1, 2, ..., L-1, 0.
  This uses path edges (i, i+1) for i=0..L-2 and chord (L-1, 0) which exists for L-1 in 2..20.
  For L=3: vertices [0,1,2], edges (0,1),(1,2),(2,0). Chord (2,0) exists.
  For L=21: vertices [0,1,...,20], edges (0,1)..(19,20),(20,0). Chord (20,0) exists.

- For L in 22..39: cycle is 0, 39, 38, ..., k, 0 where k = 41 - L.
  Path: 0 -> 39 -> 38 -> ... -> k. The edges are (0,39), (39,38), ..., (k+1, k).
  Then chord (k, 0). We need chord (k,0) to exist, i.e., k in 2..20.
  For L=22: k=19. Path 0->39->...->19 (20 edges? no: 39-19+1=21 vertices, 20 edges on path + 1 chord = 21? Wait.)
  
  Let me recount. Cycle length L means L vertices and L edges.
  For L=22, k=41-22=19. Vertices: 0, 39, 38, ..., 19. Count: 39-19+1 = 21 vertices. That's 21, not 22.
  
  Let me redo: vertices are 0, 39, 38, ..., k. Number of vertices = 1 + (39 - k) = 40 - k.
  We need 40 - k = L, so k = 40 - L.
  
  For L=22: k=18. Vertices: 0,39,38,...,18. Count: 40-18=22. Good.
  Edges on path: (0,39), (39,38), ..., (19,18). That's 21 edges. Plus chord (18,0). Total 22 edges.
  Chord (18,0) exists since 18 is in 2..20.
  
  For L=39: k=1. But chord (1,0) is already in the Hamiltonian cycle, not a chord. 
  Wait, k=1 means we need edge (1,0) which is part of the base cycle. But the path is 0->39->...->1, then edge (1,0).
  Vertices: 0,39,38,...,1. Count: 40-1=39. Good.
  The last edge is (1,0) which exists in the base cycle. So this works without needing a chord.
  
  For L=38: k=2. Vertices: 0,39,...,2. Count: 38. Edges: (0,39)..(3,2) and (2,0). Chord (2,0) exists.
  
  So for L in 22..39: k = 40 - L, which ranges from 18 down to 1.
  When k >= 2: we need chord (k,0), which exists for k in 2..20. Since k <= 18, all good.
  When k = 1: we use edge (1,0) from the base cycle.
  
  For L=40: the full Hamiltonian cycle 0,1,2,...,39,0.
"""

import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin
    
    # Base Hamiltonian cycle edges
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Chords from 0 to 2..20
    for k in range(2, 21):
        edges.add((0, k))
    
    edge_list = list(edges)
    
    # Build cycles for lengths 3..40
    cycles = []
    
    for L in range(3, N + 1):
        if time.time() > deadline:
            # Should not happen for this deterministic construction, but guard anyway
            break
        
        if L <= 21:
            # Cycle: 0, 1, 2, ..., L-1, 0
            cycle = list(range(L))
        elif L <= 39:
            # Cycle: 0, 39, 38, ..., k, 0 where k = 40 - L
            k = 40 - L
            cycle = [0] + list(range(39, k - 1, -1))
        else:  # L == 40
            cycle = list(range(N))
        
        cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
