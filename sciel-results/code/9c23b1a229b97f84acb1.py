"""
Gallai coloring of K_16 maximizing distinct edge colors.

Uses a recursive partition (cut) tree. Each internal node splits its vertex set
into two parts; all cross edges between the parts receive one color. Leaves
are small vertex sets whose internal edges are colored with distinct colors.

For n=16, we search over partition tree structures to maximize distinct colors.
"""
from itertools import combinations
import time

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


def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True


def _count_colors(colors):
    """Count distinct colors used."""
    return len(set(colors))


def _build_partition_tree(partition_spec, vertices):
    """
    Build a partition tree from a spec.
    partition_spec: list of (part1_indices, part2_indices) at each internal node,
    plus leaf assignments.
    
    Returns: list of (color_id, [(u,v), ...]) for each color group.
    """
    pass


def _try_partition(leaf_sizes, tree_structure, vertex_order):
    """
    Try a specific partition structure.
    leaf_sizes: list of sizes of leaves (should sum to N)
    tree_structure: how to combine leaves into a binary tree
    vertex_order: permutation of vertices assigned to leaves
    
    Returns a coloring or None if invalid.
    """
    colors = [-1] * NUM_EDGES
    color_counter = 0
    
    # Assign vertices to leaves
    leaves = []
    start = 0
    for size in leaf_sizes:
        leaves.append(vertex_order[start:start + size])
        start += size
    
    # Color leaf internals with distinct colors
    for leaf in leaves:
        for i in range(len(leaf)):
            for j in range(i + 1, len(leaf)):
                u, v = leaf[i], leaf[j]
                colors[_edge_idx(u, v)] = color_counter
                color_counter += 1
    
    # Build tree and color internal nodes
    # tree_structure: list of (left_indices, right_indices) where indices refer to leaf groups
    # We build a binary tree over the leaves
    
    # For a simple approach: use a balanced or specified tree
    # tree_structure is a list of (left_set, right_set) of leaf indices
    for left_leaves, right_leaves in tree_structure:
        part1 = []
        for li in left_leaves:
            part1.extend(leaves[li])
        part2 = []
        for ri in right_leaves:
            part2.extend(leaves[ri])
        
        c = color_counter
        color_counter += 1
        for u in part1:
            for v in part2:
                colors[_edge_idx(u, v)] = c
    
    # Check all edges are colored
    for c in colors:
        if c < 0:
            return None, 0
    
    # Validate
    if not _is_valid(colors):
        return None, 0
    
    return colors, _count_colors(colors)


def _generate_tree_structures(num_leaves):
    """Generate various binary tree structures for combining leaves."""
    structures = []
    
    # Balanced tree (like a tournament bracket)
    def balanced_tree(n):
        """Return list of (left_leaf_set, right_leaf_set) for balanced binary tree over n leaves."""
        if n <= 1:
            return []
        mid = n // 2
        left = list(range(mid))
        right = list(range(mid, n))
        result = [(left, right)]
        # Recursively handle subtrees? No, we just need the top-level cuts.
        # Actually for a partition tree, we need ALL internal node cuts.
        # Let me rethink: a binary tree with n leaves has n-1 internal nodes.
        # Each internal node defines a cut.
        # The tree structure determines which leaves are in which subtree.
        
        # For a balanced tree:
        # Level 0: split into left half and right half
        # Level 1: split each half, etc.
        
        cuts = []
        # Build the tree recursively
        def build_tree(leaf_indices):
            if len(leaf_indices) <= 1:
                return
            mid = len(leaf_indices) // 2
            left_idx = leaf_indices[:mid]
            right_idx = leaf_indices[mid:]
            cuts.append((left_idx, right_idx))
            build_tree(left_idx)
            build_tree(right_idx)
        
        build_tree(list(range(n)))
        return cuts
    
    structures.append(balanced_tree(num_leaves))
    
    # Left-deep tree: ((L0,L1),L2),L3),...)
    left_deep = []
    current = [0]
    for i in range(1, num_leaves):
        current.append(i)
        left_deep.append(([i], current[:-1]))  # Wait, this isn't right either
    # Let me redo: left-deep means we combine one at a time
    # Node 1: L0 vs L1 -> {L0,L1}
    # Node 2: {L0,L1} vs L2 -> {L0,L1,L2}
    # Node 3: {L0,L1,L2} vs L3 -> {L0,L1,L2,L3}
    left_deep = []
    acc = [0]
    for i in range(1, num_leaves):
        left_deep.append((acc, [i]))
        acc.append(i)
    structures.append(left_deep)
    
    # Right-deep tree
    right_deep = []
    acc = [num_leaves - 1]
    for i in range(num_leaves - 2, -1, -1):
        right_deep.append(([i], acc))
        acc.append(i)
    structures.append(right_deep)
    
    return structures


def construct():
    """
    Construct a Gallai coloring of K_16 maximizing distinct edge colors.
    
    Tries multiple partition tree structures and returns the best valid one.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    best_colors = None
    best_count = 0
    
    # Try various leaf size combinations
    # The number of distinct colors = sum(C(size_i, 2) for leaves) + (num_leaves - 1)
    # For a partition tree with k leaves, there are k-1 internal node colors.
    # Leaf i of size s_i contributes C(s_i, 2) colors.
    # Total = sum(C(s_i, 2)) + (k - 1)
    
    # To maximize: we want to maximize sum(C(s_i, 2)) + k - 1 subject to sum(s_i) = 16, s_i >= 1
    # C(s,2) = s(s-1)/2
    # For fixed sum, sum of s(s-1)/2 is maximized when sizes are as unequal as possible
    # But we also get +k-1, so more leaves is better for the linear term.
    # Trade-off: s=3 gives 3, s=2 gives 1, s=1 gives 0
    # 5 leaves of 3 + 1 leaf of 1: 5*3 + 5-1 = 15+4 = 19? Wait, that's not right.
    # 5 leaves of size 3: 5*3 = 15, 1 leaf of size 1: 0. Total leaf colors = 15.
    # 6 leaves total, so 5 internal colors. Total = 20.
    
    # Let's enumerate:
    # k leaves, sizes s_1,...,s_k, sum = 16
    # Score = sum(s_i*(s_i-1)/2) + (k-1)
    
    candidates = []
    
    # All partitions of 16 into positive integers
    # This is manageable for 16
    def gen_partitions(n, max_part=None):
        if max_part is None:
            max_part = n
        if n == 0:
            yield []
            return
        for first in range(min(n, max_part), 0, -1):
            for rest in gen_partitions(n - first, first):
                yield [first] + rest
    
    for parts in gen_partitions(16):
        k = len(parts)
        leaf_colors = sum(s * (s - 1) // 2 for s in parts)
        total = leaf_colors + (k - 1)
        candidates.append((total, parts))
    
    # Sort by total descending
    candidates.sort(key=lambda x: -x[0])
    
    # Try the top candidates
    tried_structures = set()
    
    for target_total, parts in candidates:
        if target_total <= best_count:
            break
        
        if time.time() > deadline:
            break
        
        # Try different tree structures for this leaf size configuration
        k = len(parts)
        tree_structures = _generate_tree_structures(k)
        
        for ts in tree_structures:
            ts_key = tuple(tuple(sorted(l)) for l, r in ts for l, r in [(l, r)])
            if ts_key in tried_structures:
                continue
            tried_structures.add(ts_key)
            
            # Try a few vertex orderings
            import random
            rng = random.Random(hash((tuple(parts), ts_key)) % (2**32))
            
            for trial in range(5):
                if time.time() > deadline:
                    break
                
                vertex_order = list(range(N))
                if trial > 0:
                    rng.shuffle(vertex_order)
                
                colors, count = _try_partition(parts, ts, vertex_order)
                
                if colors is not None and count > best_count:
                    best_count = count
                    best_colors = colors
    
    # If we found a good solution, return it
    if best_colors is not None:
        return best_colors
    
    # Fallback: use the original known-good construction
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
    color_node(L0, L1)
    color_node(L2, L3)
    color_node(L0 + L1, L2 + L3)
    color_node(L4, L5)
    color_node(L0 + L1 + L2 + L3, L4 + L5)

    return colors
