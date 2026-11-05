"""Improved pancyclic graph on 40 vertices using a ladder structure."""
N = 40

def construct():
    # Build a ladder graph with vertices 0..19 (top) and 20..39 (bottom)
    # Top path: (i, i+1) for i=0..18
    # Bottom path: (i+20, i+21) for i=0..18
    # Rungs: (i, i+20) for i=0..19
    # One extra chord: (0, 21) to create the triangle
    
    edges = []
    
    # Top path
    for i in range(19):
        edges.append((i, i+1))
    
    # Bottom path
    for i in range(19):
        edges.append((i+20, i+21))
    
    # Rungs
    for i in range(20):
        edges.append((i, i+20))
    
    # Extra chord to create triangle: (0, 21)
    edges.append((0, 21))
    
    # Now construct cycles of every length 3 to 40
    cycles = []
    
    for k in range(3, 41):
        if k % 2 == 1:
            # Odd cycle: use the chord (0, 21)
            # Pattern: 0-1-2-...-m-(m+20)-(m+19)-...-21-0
            # Length = m + 1 + (20 - (m+19)) + 1 = m + 1 + 1 + 1 = m + 3? 
            # Let me recalculate:
            # Path 0 to m: m edges (vertices 0,1,...,m)
            # Rung m to m+20: 1 edge
            # Path m+20 down to 21: (m+20) - 21 = m-1 edges
            # Chord 21 to 0: 1 edge
            # Total edges: m + 1 + (m-1) + 1 = 2m + 1
            # So 2m+1 = k, m = (k-1)/2
            m = (k - 1) // 2
            # Verify: k=3, m=1; k=39, m=19
            # Cycle: 0, 1, ..., m, m+20, m+19, ..., 21, 0
            cyc = list(range(0, m+1))
            cyc.extend(range(m+20, 20, -1))
            cycles.append(cyc)
        else:
            # Even cycle: use the ladder structure without the chord
            # Pattern: 0-1-...-m-(m+20)-(m+19)-...-20-0
            # Path 0 to m: m edges
            # Rung m to m+20: 1 edge
            # Path m+20 down to 20: m edges
            # Rung 20 to 0: 1 edge
            # Total: m + 1 + m + 1 = 2m + 2
            # So 2m+2 = k, m = k/2 - 1
            m = k // 2 - 1
            # Verify: k=4, m=1; k=40, m=19
            # Cycle: 0, 1, ..., m, m+20, m+19, ..., 20, 0
            cyc = list(range(0, m+1))
            cyc.extend(range(m+20, 19, -1))
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}
