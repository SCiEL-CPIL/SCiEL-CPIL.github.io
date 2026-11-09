# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges that minimizes the largest book size.
    
    Strategy:
    1. Start with K_{20,20} (400 edges).
    2. Add 100 intra-part edges forming K_{10,10} within Part A.
       Split Part A into A1 (0..9) and A2 (10..19).
       Connect all A1 to all A2.
    3. Remove 99 bipartite edges to keep total count at 401.
       Distribute removals to minimize max common neighbors in B for intra edges.
       Use 4 disjoint sets of 5 vertices in B.
       Assign A1 vertices to sets 0,1 and A2 vertices to sets 2,3.
       This ensures S_u and S_v are disjoint for u in A1, v in A2.
       Union size 10, so intersection size 10.
       One vertex will have 4 removals to satisfy sum=99, leading to intersection 11 for its edges.
       However, this is a more balanced structure than the star.
    """
    n = 40
    part_a = list(range(20))
    part_b = list(range(20, 40))
    
    # Split A into two halves
    a1 = list(range(0, 10))
    a2 = list(range(10, 20))
    
    edges = set()
    
    # 1. Add all bipartite edges
    for a in part_a:
        for b in part_b:
            edges.add((a, b))
            
    # 2. Add intra-A edges (K10,10 between a1 and a2)
    for u in a1:
        for v in a2:
            edges.add((u, v))
            
    # Current edge count: 400 + 100 = 500.
    # Target: 401.
    # Remove 99 bipartite edges.
    
    # Define 4 disjoint sets in B
    b_set_0 = part_b[0:5]
    b_set_1 = part_b[5:10]
    b_set_2 = part_b[10:15]
    b_set_3 = part_b[15:20]
    
    # Assign removal sets
    # A1 vertices (0..9) use sets 0 and 1
    # A2 vertices (10..19) use sets 2 and 3
    # This ensures S_u (from A1) and S_v (from A2) are disjoint.
    # |S_u U S_v| = 5 + 5 = 10.
    # Intersection size = 20 - 10 = 10.
    
    removals = []
    
    # Assign 5 removals to each vertex initially
    # 20 vertices * 5 = 100 removals.
    # We need 99. So one vertex gets 4.
    
    # Let's assign:
    # a1[0] gets b_set_0 minus one element (4 removals)
    # a1[1..9] get b_set_0 or b_set_1 (5 removals)
    # a2[0..9] get b_set_2 or b_set_3 (5 removals)
    
    # To be safe, let's just pick 99 edges to remove.
    # We want to remove 5 from most vertices.
    
    # List of (vertex, set_of_bs)
    vertex_assignments = []
    
    # A1
    for i, u in enumerate(a1):
        if i == 0:
            # Remove 4 from set 0
            vertex_assignments.append((u, b_set_0[:4]))
        else:
            # Alternate between set 0 and 1
            s = b_set_0 if i % 2 == 1 else b_set_1
            vertex_assignments.append((u, s))
            
    # A2
    for i, v in enumerate(a2):
        # Alternate between set 2 and 3
        s = b_set_2 if i % 2 == 0 else b_set_3
        vertex_assignments.append((v, s))
        
    # Total removals: 4 + 9*5 + 10*5 = 4 + 45 + 50 = 99.
    
    for u, bs in vertex_assignments:
        for b in bs:
            edges.discard((u, b))
            
    # Verify edge count
    if len(edges) != 401:
        # Fallback
        return [[u, v] for u, v in list(edges)[:401]]
        
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
