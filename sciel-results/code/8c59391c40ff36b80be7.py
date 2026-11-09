# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum number of edges.
    
    The minimum number of edges for a 3-chromatic 3-uniform hypergraph is 7,
    achieved by the Fano plane. We construct the Fano plane on vertices 0-6
    and leave vertices 7-8 isolated.
    
    Vertices 0-6 correspond to non-zero vectors in GF(2)^3.
    Edges are triples {u, v, u+v} for u != v, u != 0, v != 0, u != v.
    To avoid duplicates, we generate unique sets.
    
    A valid 3-coloring is provided for vertices 0-8.
    """
    # Define the 7 non-zero vectors of GF(2)^3 as integers 1..7
    # 1: 001, 2: 010, 3: 011, 4: 100, 5: 101, 6: 110, 7: 111
    # We map these to indices 0..6
    # idx 0 -> vec 1
    # idx 1 -> vec 2
    # ...
    # idx 6 -> vec 7
    
    # Generate edges
    edges = []
    seen = set()
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            if k == 0 or k <= j:
                continue
            # Ensure i, j, k are distinct and non-zero
            # In GF(2)^3, i^j=k implies i^k=j and j^k=i
            # So {i,j,k} is a line.
            # We want unique lines.
            # Sort the tuple to use as a set key
            line = tuple(sorted((i, j, k)))
            if line not in seen:
                seen.add(line)
                # Convert to 0-based indices
                edges.append([line[0] - 1, line[1] - 1, line[2] - 1])
    
    # The Fano plane has 7 edges.
    # Verify we have 7
    assert len(edges) == 7, f"Expected 7 edges, got {len(edges)}"
    
    # Define a valid 3-coloring for vertices 0..8
    # Vertices 0..6 are the Fano plane vertices.
    # Coloring derived from analysis:
    # v0 (vec 1): 0
    # v1 (vec 2): 0
    # v2 (vec 3): 1
    # v3 (vec 4): 1
    # v4 (vec 5): 2
    # v5 (vec 6): 2
    # v6 (vec 7): 0
    # v7: 0
    # v8: 0
    
    colors = [0, 0, 1, 1, 2, 2, 0, 0, 0]
    
    # Verification (optional but safe)
    for edge in edges:
        c0, c1, c2 = colors[edge[0]], colors[edge[1]], colors[edge[2]]
        if c0 == c1 == c2:
            raise ValueError(f"Edge {edge} is monochromatic with colors {colors}")
            
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
