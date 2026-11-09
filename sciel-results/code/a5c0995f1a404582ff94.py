"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a valid 4-coloring of K_50 with no monochromatic triangles.
    Uses a 5x10 block construction based on the Petersen graph.
    
    1. Partition 50 vertices into 5 blocks of 10.
    2. Within each block, use the Petersen graph (triangle-free) for Color 0 
       and its complement (triangle-free) for Color 1.
    3. Between blocks, use a 5-cycle structure on the block indices.
       Edges between blocks in the cycle get Color 2.
       Edges between blocks not in the cycle get Color 3.
       The meta-graph for Color 2 is C_5 (triangle-free).
       The meta-graph for Color 3 is C_5 (triangle-free).
    4. Mixed triangles (2 vertices in one block, 1 in another) cannot be 
       monochromatic because internal edges use {0,1} and external edges use {2,3}.
    """
    n = 50
    block_size = 10
    num_blocks = 5
    
    # Define Petersen graph edges for a block of 10 vertices (0..9)
    # Petersen graph is the complement of the line graph of K_5, or 
    # can be defined by vertices (i, j) where 0 <= i < j <= 4.
    # Let's map 0..9 to pairs (i,j):
    # 0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4), 4: (1,2), 5: (1,3), 6: (1,4), 7: (2,3), 8: (2,4), 9: (3,4)
    # Two vertices are connected in Petersen if their pairs are disjoint.
    def is_petersen_edge(u, v):
        # u, v in 0..9
        # Map to pairs
        pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (3, 4)
        ]
        pu = pairs[u]
        pv = pairs[v]
        # Check if disjoint
        return len(set(pu) & set(pv)) == 0

    # Precompute Petersen adjacency for local indices
    petersen_adj = [[False]*10 for _ in range(10)]
    for i in range(10):
        for j in range(i+1, 10):
            if is_petersen_edge(i, j):
                petersen_adj[i][j] = True
                petersen_adj[j][i] = True

    # Define the 5-cycle on block indices
    # Edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    cycle_edges = {
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
        (3, 4), (4, 3),
        (4, 0), (0, 4)
    }

    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            b_i = i // block_size
            b_j = j // block_size
            u = i % block_size
            v = j % block_size
            
            if b_i == b_j:
                # Internal edge: Color 0 if Petersen, 1 if not
                if petersen_adj[u][v]:
                    colors.append(0)
                else:
                    colors.append(1)
            else:
                # External edge: Color 2 if block edge in cycle, 3 if not
                if (b_i, b_j) in cycle_edges:
                    colors.append(2)
                else:
                    colors.append(3)
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
