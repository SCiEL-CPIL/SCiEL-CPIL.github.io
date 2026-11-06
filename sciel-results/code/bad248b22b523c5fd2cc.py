"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Ladder graph on 40 vertices (2 rows of 20) + 1 diagonal chord
    # Row 0: 0..19, Row 1: 20..39
    # Edges: row-edges + rungs
    # Then 1 chord to break bipartiteness (gives all odd cycles >= 3)
    # Ladder has all even cycles 4..40
    # Chord (0, 21) creates triangle 0-1-21-0 (wait, need to check)
    # Actually chord (0, 21): 0 is in row 0, 21 is in row 1.
    # 0-20 is a rung, 20-21 is a row edge, so 0-20-21 is a path.
    # 0-21 is the chord. So 0-20-21-0 is a triangle. Good.
    # Now we have odd cycle 3.
    # For odd cycles 5,7,...,39: use the chord to shortcut.
    # For even cycles 4,6,...,40: use the ladder's natural cycles.
    
    edges = []
    
    # Row 0 edges: (i, i+1) for i in 0..18
    for i in range(19):
        edges.append((i, i + 1))
    
    # Row 1 edges: (20+i, 21+i) for i in 0..18
    for i in range(19):
        edges.append((20 + i, 21 + i))
    
    # Rungs: (i, 20+i) for i in 0..19
    for i in range(20):
        edges.append((i, 20 + i))
    
    # Chord to break bipartiteness: (0, 21)
    edges.append((0, 21))
    
    # Total edges: 19 + 19 + 20 + 1 = 59
    # Score = 59 - 40 = 19
    
    # Now construct cycles for lengths 3..40
    cycles = []
    
    for k in range(3, N + 1):
        if k == 3:
            # Triangle: 0 - 20 - 21 - 0
            cycles.append([0, 20, 21])
        elif k % 2 == 1:
            # Odd cycle of length k >= 5
            # Use chord (0, 21) and go along row 0 then row 1
            # Path: 0 -> 1 -> 2 -> ... -> j -> 20+j -> 20+j-1 -> ... -> 21 -> 0
            # Length: (j+1) edges from 0 to j, then (j+1 - 1) edges from 20+j to 21, then chord
            # Wait: 0 to j is j edges. 20+j to 21 is (20+j - 21) = j-1 edges. Then chord 21->0 is 1 edge.
            # Total: j + (j-1) + 1 = 2j edges. But we want odd.
            # Let me reconsider.
            # 0 -> 1 -> ... -> j (j edges), j -> 20+j (1 edge, rung), 20+j -> 20+j-1 -> ... -> 21 (j-1 edges), 21 -> 0 (1 edge, chord)
            # Total: j + 1 + (j-1) + 1 = 2j + 1 edges. 
            # For length k: 2j+1 = k, so j = (k-1)/2
            j = (k - 1) // 2
            # Vertices: 0, 1, ..., j, 20+j, 20+j-1, ..., 21
            cycle = list(range(0, j + 1)) + list(range(20 + j, 20, -1))
            cycles.append(cycle)
        else:
            # Even cycle of length k >= 4
            # Use a rectangular cycle in the ladder
            # 0 -> 1 -> ... -> j -> 20+j -> 20+j-1 -> ... -> 20 -> 0
            # Length: j + 1 + j + 1 = 2(j+1) edges
            # For length k: 2(j+1) = k, so j = k/2 - 1
            j = k // 2 - 1
            # Vertices: 0, 1, ..., j, 20+j, 20+j-1, ..., 20
            cycle = list(range(0, j + 1)) + list(range(20 + j, 19, -1))
            cycles.append(cycle)
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
