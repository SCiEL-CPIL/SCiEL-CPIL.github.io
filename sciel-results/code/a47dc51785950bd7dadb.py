"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 8 disjoint copies of K4.
    # n = 32
    # Each K4 has 4 vertices and 6 edges.
    # Total edges = 8 * 6 = 48.
    # No C5 (max cycle in K4 is C4? No, K4 has C3. Does K4 have C4?
    # Vertices 0,1,2,3. 0-1-2-3-0 is a cycle of length 4.
    # Wait, the task says "no 5-cycle". C4 is allowed.
    # K4 contains C3 and C4. It does NOT contain C5.
    # Independence number of 8 disjoint K4s is 8 (pick one from each).
    # We need no independent set of 9. 8 < 9. OK.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        # Vertices: base, base+1, base+2, base+3
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
