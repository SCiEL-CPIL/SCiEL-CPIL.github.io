"""Combined program: deterministic Gallai coloring achieving the proven maximum of 20 distinct colors."""
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Gallai coloring of K_16 maximizing distinct edge colors.
    
    Uses a fixed recursive partition structure with five 3-vertex leaves and one 1-vertex leaf.
    This achieves exactly 20 distinct colors, which is the provable maximum for n=16.
    
    Structure:
    - Leaves: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15]
    - Each 3-vertex leaf has 3 edges, all with distinct colors (5*3 = 15 colors)
    - Inter-part edges at each internal node are monochromatic (5 internal nodes = 5 colors)
    - Total: 15 + 5 = 20 distinct colors
    
    The binary tree structure ensures no 4-cycle has four distinct edge colors.
    """
    colors = [0] * NUM_EDGES
    color_counter = [0]

    def get_new_color():
        c = color_counter[0]
        color_counter[0] += 1
        return c

    def color_all_distinct(vertices):
        """Color all edges within vertices with distinct colors."""
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                colors[_edge_idx(u, v)] = get_new_color()

    def color_node(part1, part2):
        """Color inter-part edges between part1 and part2 with one new color."""
        c = get_new_color()
        for u in part1:
            for v in part2:
                colors[_edge_idx(u, v)] = c

    # Fixed partition: five 3-vertex leaves + one 1-vertex leaf
    leaves = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15]
    ]

    # Color leaf internals with distinct colors
    for leaf in leaves:
        if len(leaf) >= 2:
            color_all_distinct(leaf)

    # Binary tree structure over the 6 leaves:
    #   Node0: L0 vs L1  -> [0..5]
    #   Node1: L2 vs L3  -> [6..11]
    #   Node2: Node0 vs Node1  -> [0..11]
    #   Node3: L4 vs L5  -> [12..15]
    #   Root:  Node2 vs Node3  -> [0..15]

    # Node0: color edges between leaves[0] and leaves[1]
    color_node(leaves[0], leaves[1])
    
    # Node1: color edges between leaves[2] and leaves[3]
    color_node(leaves[2], leaves[3])
    
    # Node2: color edges between (L0+L1) and (L2+L3)
    part2a = leaves[0] + leaves[1]
    part2b = leaves[2] + leaves[3]
    color_node(part2a, part2b)
    
    # Node3: color edges between leaves[4] and leaves[5]
    color_node(leaves[4], leaves[5])
    
    # Root: color edges between (L0+L1+L2+L3) and (L4+L5)
    part_root_a = leaves[0] + leaves[1] + leaves[2] + leaves[3]
    part_root_b = leaves[4] + leaves[5]
    color_node(part_root_a, part_root_b)

    # All 20 distinct colors have been assigned.
    # This is a valid Gallai coloring by construction.
    return colors
# EVOLVE-BLOCK-END
