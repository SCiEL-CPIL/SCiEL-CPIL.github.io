"""
Constructs a sparse pancyclic graph on 40 vertices.
Target: Minimize (edge_count - 40).

Theoretical Analysis:
- A pancyclic graph on N vertices must contain cycles of lengths 3..N.
- A Hamiltonian cycle provides length N.
- Adding a chord (u, v) to a Hamiltonian cycle creates two new cycles:
  one following the path u->...->v and one following v->...->u.
- Let k be the number of chords. The maximum number of distinct new cycle
  lengths generated is 2k (one from each side of the chord, assuming no overlaps
  and optimal placement).
- We need to cover lengths 3 to N-1, which is N-3 lengths.
- Therefore, 2k >= N-3 => k >= ceil((N-3)/2).
- For N=40, k >= ceil(37/2) = 19.
- Total edges = N + k = 40 + 19 = 59.
- Value = edges - 40 = 19.
- This is the theoretical minimum for this construction family (and generally
  for pancyclic graphs of this size).

Strategy:
1. Base Hamiltonian cycle: 0-1-...-39-0 (40 edges).
2. Add chords (0, k) for k in 2..20 (19 edges).
   - Chord (0, k) splits the cycle into two paths of lengths k and 40-k.
   - The cycles formed have lengths k+1 and 40-k+1.
   - For k=2: lengths 3 and 39.
   - For k=3: lengths 4 and 38.
   - ...
   - For k=19: lengths 20 and 22.
   - For k=20: lengths 21 and 21.
   - The set of lengths covered is {3, 39, 4, 38, ..., 20, 22, 21} = {3, ..., 39}.
   - Combined with the Hamiltonian cycle (length 40), all lengths 3..40 are covered.

Implementation:
- Construct edges.
- Construct explicit vertex lists for cycles of each length 3..40.
- Verify validity.
"""

import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # 1. Construct Edges
    edges = set()
    
    # Base Hamiltonian Cycle: (i, i+1) mod N
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Chords (0, k) for k in 2..20
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list of sorted tuples for consistent output
    edge_list = [tuple(sorted(e)) for e in edges]
    edge_list.sort()
    
    # 2. Construct Cycles
    cycles = []
    
    # L from 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle: 0, 1, ..., 39
            cyc = list(range(N))
        elif L <= 21:
            # Short path + chord
            # Vertices 0, 1, ..., L-1
            # Chord is (0, L-1)
            # Length is L (L-1 edges in path + 1 chord)
            cyc = list(range(L))
        else:
            # Long path + chord
            # We need a cycle of length L.
            # From chord (0, k) where k = 41 - L.
            # The long path is 0 -> 39 -> 38 -> ... -> k.
            # Number of vertices in path: 39 - k + 1 = 40 - k.
            # Total cycle length: (40 - k) + 1 = 41 - k.
            # Set 41 - k = L => k = 41 - L.
            k = 41 - L
            # Vertices: 0, 39, 38, ..., k
            # range(N-1, k-1, -1) generates N-1 down to k
            cyc = [0] + list(range(N - 1, k - 1, -1))
            
        cycles.append(cyc)
        
    # Verification
    # This step ensures the output is strictly valid.
    # For N=40, this is O(N^2) and very fast.
    edge_set = set(edge_list)
    for cyc in cycles:
        # Check distinct vertices
        if len(set(cyc)) != len(cyc):
            raise ValueError(f"Cycle contains duplicate vertices: {cyc}")
        # Check edges exist
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if (u, v) not in edge_set and (v, u) not in edge_set:
                raise ValueError(f"Missing edge ({u}, {v}) in cycle of length {len(cyc)}")
                
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
