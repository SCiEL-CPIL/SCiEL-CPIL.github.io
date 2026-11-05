"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Vertices 0..19 = part A, 20..39 = part B
    # Star of 10 internal edges in A: center=0, leaves=1..10
    # Remove 13 bipartite edges from center (0 to B[0:13])
    # Keep bipartite edges: (0, 33..39) = 7 edges, all (1..19, 20..39) = 380 edges
    # Total: 10 + 7 + 380 = 397... need 401. Add 4 more internal edges among leaves.
    
    # Actually let me recount: K20,20 has 400 edges. We need 401.
    # Remove 13 from center in A: lose 13 edges. Add 10 star + 4 extra internal = +14.
    # Net: 400 - 13 + 14 = 401. Good.
    
    edges = []
    
    # All bipartite edges except (0, 20..32)
    for a in range(20):
        for b in range(20, 40):
            if a == 0 and b < 33:  # remove 13 edges from center to B[0:13]
                continue
            edges.append([a, b])
    
    # Star: 0 connected to 1..10
    for leaf in range(1, 11):
        edges.append([0, leaf])
    
    # 4 additional internal edges among leaves (1..19) to reach 401
    # Use edges (11,12), (13,14), (15,16), (17,18)
    for u, v in [(11,12), (13,14), (15,16), (17,18)]:
        edges.append([u, v])
    
    # Verify count
    assert len(edges) == 401, f"Expected 401 edges, got {len(edges)}"
    # Verify distinct
    assert len(set(tuple(sorted(e)) for e in edges)) == 401, "Edges not distinct"
    
    return edges
# EVOLVE-BLOCK-END
