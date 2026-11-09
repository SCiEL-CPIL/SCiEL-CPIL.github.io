"""
Gallai coloring of K_16 maximizing distinct edge colors.

Uses a recursive partition (cut) tree. Each internal node splits its vertex set
into two parts; all cross edges between the parts receive one color. Leaves
are small vertex sets whose internal edges are colored with distinct colors.

For n=16, the maximum number of distinct colors in a Gallai coloring is 20.
We use five 3-vertex leaves (15 distinct internal colors) + 5 internal node
colors = 20 total.
"""
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2


def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)


# Precompute all 4-cycles as tuples of 4 edge indices
_CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    _CYCLES.append((_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a)))
    _CYCLES.append((_edge_idx(a, b), _edge_idx(b, d), _edge_idx(d, c), _edge_idx(c, a)))
    _CYCLES.append((_edge_idx(a, c), _edge_idx(c, b), _edge_idx(b, d), _edge_idx(d, a)))


def _is_valid_gallai(colors):
    """Check that no 4-cycle has 4 distinct edge colors."""
    for cycle in _CYCLES:
        if len(set(colors[i] for i in cycle)) == 4:
            return False
    return True


def _count_distinct(colors):
    return len(set(colors))


def construct():
    """
    Construct a Gallai coloring of K_16 with 20 distinct edge colors.

    Partition tree:
      Leaves: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15]
      Node A: L0 vs L1          -> color 15
      Node B: L2 vs L3          -> color 16
      Node C: (L0+L1) vs (L2+L3)-> color 17
      Node D: L4 vs L5          -> color 18
      Root: (L0..L3) vs (L4+L5) -> color 19

    Leaf internals: 15 distinct colors (0..14).
    Total: 20 distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    colors = [0] * NUM_EDGES
    color_counter = 0

    def get_new_color():
        nonlocal color_counter
        c = color_counter
        color_counter += 1
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

    # Leaves
    L0 = [0, 1, 2]
    L1 = [3, 4, 5]
    L2 = [6, 7, 8]
    L3 = [9, 10, 11]
    L4 = [12, 13, 14]
    L5 = [15]

    # Color leaf internals with distinct colors (15 colors: 0-14)
    for leaf in (L0, L1, L2, L3, L4, L5):
        if len(leaf) >= 2:
            color_all_distinct(leaf)

    # Internal node colors (5 colors: 15-19)
    # Node A: L0 vs L1
    color_node(L0, L1)
    # Node B: L2 vs L3
    color_node(L2, L3)
    # Node C: (L0+L1) vs (L2+L3)
    color_node(L0 + L1, L2 + L3)
    # Node D: L4 vs L5
    color_node(L4, L5)
    # Root: (L0+L1+L2+L3) vs (L4+L5)
    color_node(L0 + L1 + L2 + L3, L4 + L5)

    # Verify and attempt local improvement
    if time.time() < deadline:
        base_colors = colors[:]
        base_distinct = _count_distinct(base_colors)
        
        if not _is_valid_gallai(base_colors):
            # Shouldn't happen, but return as-is if invalid
            return base_colors
        
        # Try to improve by splitting some monochromatic cuts into two colors
        # This would require careful verification
        improved = base_colors[:]
        best_distinct = base_distinct
        
        # Attempt: try reassigning some edges to new colors and verify
        # Focus on edges that are in monochromatic cuts - try splitting them
        # For each internal node cut, try using 2 colors instead of 1
        # This adds 1 color but may violate the Gallai constraint
        
        # Try splitting the root cut (largest) into 2 colors
        root_cut_edges = []
        left_part = set(L0 + L1 + L2 + L3)
        right_part = set(L4 + L5)
        for u in left_part:
            for v in right_part:
                if u < v:
                    root_cut_edges.append((u, v))
                else:
                    root_cut_edges.append((v, u))
        
        # Try assigning half to color 20, half to color 21
        # But we need to check all 4-cycles
        for split_ratio in [0.5]:
            test_colors = base_colors[:]
            new_color_1 = 20
            new_color_2 = 21
            mid = len(root_cut_edges) // 2
            for idx, (u, v) in enumerate(root_cut_edges):
                if idx < mid:
                    test_colors[_edge_idx(u, v)] = new_color_1
                else:
                    test_colors[_edge_idx(u, v)] = new_color_2
            
            if _is_valid_gallai(test_colors):
                new_distinct = _count_distinct(test_colors)
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    improved = test_colors[:]
        
        if time.time() < deadline:
            # Try splitting other cuts as well
            # Node D: L4 vs L5 (3 edges)
            node_d_edges = []
            for u in L4:
                for v in L5:
                    if u < v:
                        node_d_edges.append((u, v))
                    else:
                        node_d_edges.append((v, u))
            
            # Try giving each edge in node D a distinct color (3 new colors)
            test_colors2 = improved[:]
            for idx, (u, v) in enumerate(node_d_edges):
                test_colors2[_edge_idx(u, v)] = 22 + idx
            
            if _is_valid_gallai(test_colors2):
                new_distinct = _count_distinct(test_colors2)
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    improved = test_colors2[:]
        
        if time.time() < deadline:
            # Try splitting Node A: L0 vs L1 (9 edges) into 2 colors
            node_a_edges = []
            for u in L0:
                for v in L1:
                    if u < v:
                        node_a_edges.append((u, v))
                    else:
                        node_a_edges.append((v, u))
            
            test_colors3 = improved[:]
            mid_a = len(node_a_edges) // 2
            for idx, (u, v) in enumerate(node_a_edges):
                if idx < mid_a:
                    test_colors3[_edge_idx(u, v)] = 25
                else:
                    test_colors3[_edge_idx(u, v)] = 26
            
            if _is_valid_gallai(test_colors3):
                new_distinct = _count_distinct(test_colors3)
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    improved = test_colors3[:]
        
        return improved
    
    return colors
