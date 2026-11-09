"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a coloring of K_16 maximizing distinct edge colors,
    subject to no four-cycle having four distinct edge colors.
    Uses a hierarchical structure with 3-way splits to maximize colors.
    """
    
    # Precompute edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    
    colors = [0] * (N*(N-1)//2)
    color_counter = [0]
    
    def get_new_color():
        c = color_counter[0]
        color_counter[0] += 1
        return c

    def color_edges_between(part1, part2, color):
        """Color all edges between part1 and part2 with given color."""
        for u in part1:
            for v in part2:
                a, b = min(u, v), max(u, v)
                colors[edge_idx[(a, b)]] = color

    def color_all_distinct(vertices):
        """Color all edges within vertices with distinct colors."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if u > v:
                    u, v = v, u
                colors[edge_idx[(u, v)]] = get_new_color()

    # Partition into 5 leaves of size 3 and 1 leaf of size 1
    leaves = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15]
    ]
    
    # Color leaf internals
    for leaf in leaves:
        if len(leaf) >= 2:
            color_all_distinct(leaf)
            
    # Tree Structure:
    # Node A: leaves[0], leaves[1], leaves[2] (3-way split)
    # Node B: leaves[3], leaves[4], leaves[5] (3-way split)
    # Root: Node A, Node B (2-way split)
    
    # Node A: 3-way split
    # Colors for cuts (0,1), (1,2), (0,2)
    c_A01 = get_new_color()
    c_A12 = get_new_color()
    c_A02 = get_new_color()
    
    color_edges_between(leaves[0], leaves[1], c_A01)
    color_edges_between(leaves[1], leaves[2], c_A12)
    color_edges_between(leaves[0], leaves[2], c_A02)
    
    # Node B: 3-way split
    # Colors for cuts (3,4), (4,5), (3,5)
    c_B34 = get_new_color()
    c_B45 = get_new_color()
    c_B35 = get_new_color()
    
    color_edges_between(leaves[3], leaves[4], c_B34)
    color_edges_between(leaves[4], leaves[5], c_B45)
    color_edges_between(leaves[3], leaves[5], c_B35)
    
    # Root: 2-way split between Node A and Node B
    part_A = leaves[0] + leaves[1] + leaves[2]
    part_B = leaves[3] + leaves[4] + leaves[5]
    
    c_Root = get_new_color()
    color_edges_between(part_A, part_B, c_Root)
    
    return colors

# EVOLVE-BLOCK-END
