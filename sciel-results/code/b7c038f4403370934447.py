"""
Constructs a sparse pancyclic graph on 40 vertices.
Goal: Minimize (edge_count - 40).

Strategy:
1. Base Hamiltonian cycle: 0-1-...-39-0 (40 edges).
2. Add chords from vertex 0 to vertices 2..19 (18 edges).
   Total edges: 58. Score: 18.

Cycles:
- L=40: Base cycle [0, 1, ..., 39].
- L in 3..19: Cycle [0, 1, ..., L-1]. Uses path edges (i, i+1) and chord (L-1, 0).
  Note: For L=3, chord is (2,0). For L=19, chord is (18,0).
- L=20: Cycle [0, 1, ..., 19]. Uses path edges and chord (19,0).
  Wait, my chords are 2..19. So chord (19,0) exists.
  So L=20 is covered by [0, 1, ..., 19].
- L in 21..39:
  We need a cycle of length L.
  Consider the path 0 -> 39 -> 38 -> ... -> k.
  Length of this path in vertices is 40 - k.
  We close it with chord (k, 0).
  We need 40 - k = L => k = 40 - L.
  For L=21, k=19. Chord (19,0) exists. Cycle: [0, 39, 38, ..., 19].
  For L=39, k=1. Chord (1,0) is a base edge. Cycle: [0, 39, 38, ..., 1].
  Wait, for L=39, k=1. The path is 0-39-...-1. Vertices: 0, 39, ..., 1. Count: 39.
  Edges: (0,39) base, (39,38) base, ..., (2,1) base. Closing edge (1,0) base.
  This works.
  
  Check L=20 again.
  If we use the "long way" formula: k = 40-20 = 20. Chord (20,0) does NOT exist in my set (2..19).
  So we must use the "short way" for L=20.
  Short way: [0, 1, ..., 19]. Closing edge (19,0). Chord (19,0) exists.
  
  So:
  L in 3..20: Use [0, 1, ..., L-1]. Requires chord (L-1, 0).
  L-1 ranges from 2 to 19. Chords 2..19 exist.
  
  L in 21..39: Use [0, 39, 38, ..., k] where k = 40-L.
  k ranges from 19 down to 1.
  Closing edge (k, 0).
  If k=19, chord (19,0) exists.
  If k=18, chord (18,0) exists.
  ...
  If k=2, chord (2,0) exists.
  If k=1, edge (1,0) is base edge.
  
  All cycles are valid.
  Total edges: 40 (base) + 18 (chords 2..19) = 58.
  Score: 58 - 40 = 18.
"""

import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Base Hamiltonian cycle
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords from 0 to k for k in 2..19
    for k in range(2, 20):
        edges.add((0, k))
    
    # Ensure edges are stored as frozensets or sorted tuples for consistency if needed, 
    # but the task asks for undirected edge pairs. (u, v) is fine as long as u!=v.
    # Let's normalize to (min, max) just in case, though the checker likely handles undirected.
    # The prompt says "undirected edge pairs". 
    # I will return a list of tuples.
    
    edge_list = list(edges)
    
    # Generate cycles
    cycles = []
    
    # L = 3 to 20
    for L in range(3, 21):
        # Cycle: 0, 1, ..., L-1
        cyc = list(range(L))
        cycles.append(cyc)
        
    # L = 21 to 39
    for L in range(21, 40):
        k = 40 - L
        # Cycle: 0, 39, 38, ..., k
        # Vertices: 0, then N-1 down to k
        cyc = [0] + list(range(N-1, k-1, -1))
        cycles.append(cyc)
        
    # L = 40
    cyc = list(range(N))
    cycles.append(cyc)
    
    # Verify within time limit (just in case, though this is static)
    # Simple check: do all edges in cycles exist in edge_list?
    # This is O(E_total * L_avg) which is small.
    # But since it's static, we can skip heavy verification if we are confident.
    # However, a bug would score 0. Let's do a quick check.
    
    # Convert to set of frozensets for O(1) lookup
    edge_set = set()
    for u, v in edge_list:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    for cyc in cycles:
        L = len(cyc)
        for i in range(L):
            u = cyc[i]
            v = cyc[(i + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                # Fallback to seed if error (should not happen)
                return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
                        'cycles': [list(range(k)) for k in range(3,N+1)]}
                        
    # Check distinct vertices in cycles
    for cyc in cycles:
        if len(set(cyc)) != len(cyc):
             return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
                     'cycles': [list(range(k)) for k in range(3,N+1)]}

    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
