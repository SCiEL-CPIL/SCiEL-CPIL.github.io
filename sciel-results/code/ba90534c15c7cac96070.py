"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    edges = []
    
    # Helper to add K3,3 edges between two sets of 3 vertices
    def add_k33(part_a, part_b):
        for u in part_a:
            for v in part_b:
                edges.append([u, v])

    # Component 1: K3,3 on 0-5
    add_k33([0, 1, 2], [3, 4, 5])
    
    # Component 2: K3,3 on 6-11
    add_k33([6, 7, 8], [9, 10, 11])
    
    return edges
# EVOLVE-BLOCK-END
