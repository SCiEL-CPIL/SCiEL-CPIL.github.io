"""
Constructs a sparse pancyclic graph on 40 vertices.
Target: Minimize (edge_count - 40).
Strategy:
1. Base Hamiltonian cycle 0-1-...-39-0 (40 edges).
2. Add chords (0, k) for k in 2..20 (19 edges).
Total edges: 59. Value: 19.
Cycles:
- L=40: Base cycle.
- L in 3..21: Path 0->1->...->(L-1) + Chord (L-1, 0). 
  Note: For L=3, path 0-1-2, chord (2,0).
  For L=21, path 0-1-...-20, chord (20,0).
- L in 22..39: Path 0->39->...->k + Chord (k, 0) where k = 41-L.
  Note: For L=22, k=19. Path 0-39-38-...-19, chord (19,0).
  Wait, distance from 0 to 19 is 19. Cycle length 19+1=20?
  Let's re-verify the "long" side.
  Chord (0, k). Distance d = k (if k < 20) or N-k.
  Cycle 1 length: d + 1.
  Cycle 2 length: N - d + 1.
  
  If we use chords (0, 2) to (0, 20):
  k=2: d=2. L1=3, L2=39.
  k=3: d=3. L1=4, L2=38.
  ...
  k=19: d=19. L1=20, L2=22.
  k=20: d=20. L1=21, L2=21.
  
  Set of lengths from chords:
  {3, 39, 4, 38, ..., 20, 22, 21}
  Union: {3, 4, ..., 21, 22, ..., 39} = {3, ..., 39}.
  Plus Hamiltonian cycle: 40.
  Total: 3..40.
  
  The cycles in the output must be vertex lists.
  For L in 3..21:
    k = L - 1.
    Cycle: [0, 1, ..., k, 0] -> [0, 1, ..., L-1]
    Wait, the cycle is 0 -> 1 -> ... -> k -> 0.
    Vertices: 0, 1, ..., k.
    Length k+1 = L.
    Edges: (0,1), (1,2), ..., (k-1, k), (k, 0).
    All these edges exist?
    (0,1)..(k-1,k) are in Hamiltonian cycle.
    (k,0) is a chord (since k is in 2..20).
    Yes.
    
  For L in 22..39:
    We need a cycle of length L.
    From the analysis, chord (0, k) with k = 41 - L creates a cycle of length N - k + 1 = 40 - (41-L) + 1 = L.
    The cycle uses the "long" path from 0 to k.
    Path: 0 -> 39 -> 38 -> ... -> k.
    Vertices: 0, 39, 38, ..., k.
    Length: (39 - k + 1) + 1 = 41 - k.
    Since k = 41 - L, length = L.
    Edges: (0,39), (39,38), ..., (k+1, k), (k, 0).
    (0,39) is in Hamiltonian cycle.
    (39,38)... are in Hamiltonian cycle.
    (k,0) is a chord.
    Yes.
    
  For L=40:
    Cycle: 0, 1, ..., 39.
    Edges: (0,1), ..., (39,0).
    All in Hamiltonian cycle.
"""

import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Construct Edges
    edges = set()
    
    # Base Hamiltonian Cycle
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list of tuples, ensuring consistent ordering for output if needed
    # The task says "undirected edge pairs", usually (u, v) with u < v or just distinct pairs.
    # I will output sorted pairs to be safe, or just the set.
    # The example output uses list of tuples.
    # Sort edges for deterministic output
    edge_list = [tuple(sorted(e)) for e in edges]
    edge_list.sort()
    
    # 2. Construct Cycles
    cycles = []
    
    # L from 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif L <= 21:
            # Short path + chord
            # Vertices 0, 1, ..., L-1
            cyc = list(range(L))
        else:
            # Long path + chord
            # k = 41 - L
            k = 41 - L
            # Vertices 0, 39, 38, ..., k
            cyc = [0] + list(range(N - 1, k - 1, -1))
            
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
