"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Gallai coloring of K_16 maximizing distinct edge colors.
    Uses recursive partition: split into parts, color inter-part edges monochromatically,
    recurse on each part. Leaves of size <= 3 get all-distinct edge colors.
    Maximum distinct colors = 20 (five 3-vertex leaves + one 1-vertex leaf).
    """
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Precompute edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    
    colors = [0] * (N*(N-1)//2)
    color_counter = [0]  # mutable counter for distinct colors used
    
    def get_new_color():
        c = color_counter[0]
        color_counter[0] += 1
        return c
    
    def color_edges_in_set(vertices, color):
        """Color all edges within vertices with given color."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if u > v:
                    u, v = v, u
                colors[edge_idx[(u, v)]] = color
    
    def color_all_distinct(vertices):
        """Color all edges within vertices with distinct colors."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if u > v:
                    u, v = v, u
                colors[edge_idx[(u, v)]] = get_new_color()
    
    def gallai_color(vertices):
        """Recursively Gallai-color the subgraph induced by vertices."""
        n = len(vertices)
        if n <= 3:
            if n >= 2:
                color_all_distinct(vertices)
            return
        
        # Split into two parts
        # For optimal: we want to end up with leaves of size 3 (and one of size 1)
        # Greedy: split as evenly as possible, but prefer splits that lead to size-3 leaves
        # Simple approach: split into two roughly equal halves
        mid = n // 2
        part1 = vertices[:mid]
        part2 = vertices[mid:]
        
        # Color inter-part edges with a single new color
        inter_color = get_new_color()
        for u in part1:
            for v in part2:
                a, b = min(u, v), max(u, v)
                colors[edge_idx[(a, b)]] = inter_color
        
        # Recurse
        gallai_color(part1)
        gallai_color(part2)
    
    # We need a smarter split to ensure we get five 3-vertex leaves and one 1-vertex leaf.
    # Let's precompute the partition structure.
    
    # Reset
    color_counter[0] = 0
    colors = [0] * (N*(N-1)//2)
    
    # Build partition tree explicitly
    # Partition: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15]
    # Tree: binary tree combining these 6 leaves
    
    # We'll build a binary tree. Let's use a balanced-ish tree.
    # Leaves: L0=[0,1,2], L1=[3,4,5], L2=[6,7,8], L3=[9,10,11], L4=[12,13,14], L5=[15]
    
    # Tree structure:
    #         Root
    #        /    \
    #      N1      N2
    #     /  \    /  \
    #    N3  N4  N5  L5
    #   / \  / \  / \
    #  L0 L1 L2 L3 L4 ... wait, that's 5 leaves on left, 1 on right.
    
    # Let me do:
    # Root splits into A=[0..8] and B=[9..15]
    # A splits into A1=[0..2] and A2=[3..8]
    # A2 splits into A2a=[3..5] and A2b=[6..8]
    # B splits into B1=[9..11] and B2=[12..15]
    # B2 splits into B2a=[12..14] and B2b=[15]
    
    # This gives 6 leaves: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15]
    
    def solve(vertices):
        """Return a partition tree as nested structure, and assign colors."""
        n = len(vertices)
        if n <= 3:
            if n >= 2:
                color_all_distinct(vertices)
            return
        
        # Choose split point to eventually get size-3 leaves
        # For n=16: split 9/7, then recurse
        # Actually, let's just do a fixed strategy: split into floor(n/3)*3 and remainder
        # Better: use the known optimal structure
        
        # Simple recursive: split into two parts, recurse
        # To get the right leaf sizes, we need careful splitting.
        # Let's use: if n % 3 == 0, split into n/3 and 2n/3 (both multiples of 3 if possible)
        # This is getting complex. Let me just use a fixed partition.
        
        pass
    
    # Fixed partition approach
    leaves = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15]
    ]
    
    # Build binary tree over these 6 leaves
    # Tree: 
    #   Node0: leaves[0] + leaves[1]  -> [0..5]
    #   Node1: leaves[2] + leaves[3]  -> [6..11]
    #   Node2: Node0 + Node1          -> [0..11]
    #   Node3: leaves[4] + leaves[5]  -> [12..15]
    #   Root: Node2 + Node3           -> [0..15]
    
    def color_node(part1, part2):
        """Color inter-part edges between part1 and part2 with one color."""
        c = get_new_color()
        for u in part1:
            for v in part2:
                a, b = min(u, v), max(u, v)
                colors[edge_idx[(a, b)]] = c
    
    # Color leaf internals (distinct colors)
    for leaf in leaves:
        if len(leaf) >= 2:
            color_all_distinct(leaf)
    
    # Color tree edges
    # Node0: leaves[0] vs leaves[1]
    color_node(leaves[0], leaves[1])
    # Node1: leaves[2] vs leaves[3]
    color_node(leaves[2], leaves[3])
    # Node2: (leaves[0]+leaves[1]) vs (leaves[2]+leaves[3])
    part2a = leaves[0] + leaves[1]
    part2b = leaves[2] + leaves[3]
    color_node(part2a, part2b)
    # Node3: leaves[4] vs leaves[5]
    color_node(leaves[4], leaves[5])
    # Root: (leaves[0..3]) vs (leaves[4]+leaves[5])
    part_root_a = leaves[0] + leaves[1] + leaves[2] + leaves[3]
    part_root_b = leaves[4] + leaves[5]
    color_node(part_root_a, part_root_b)
    
    return colors

# EVOLVE-BLOCK-END
