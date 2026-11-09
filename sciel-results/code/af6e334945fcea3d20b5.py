"""
Gallai coloring of K_16 maximizing distinct edge colors.

Uses a recursive partition (cut) tree. Each internal node splits its vertex set
into two parts; all cross edges between the parts receive one color. Leaves
are small vertex sets whose internal edges are colored with distinct colors.

For n=16, the maximum number of distinct colors in a Gallai coloring is 20.
We use five 3-vertex leaves (15 distinct internal colors) + 5 internal node
colors = 20 total.

Then we do systematic local search to try to add more colors by splitting
monochromatic cuts into multiple colors while maintaining the Gallai property.
"""
import time
from itertools import combinations, product

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


def _is_valid_gallai_fast(colors, changed_edges):
    """Fast check: only verify 4-cycles that include at least one changed edge."""
    # Build set of changed edge indices for quick lookup
    changed_set = set(changed_edges)
    
    for cycle in _CYCLES:
        # Only check if at least one edge in the cycle was changed
        if any(e in changed_set for e in cycle):
            if len(set(colors[i] for i in cycle)) == 4:
                return False
    return True


def _count_distinct(colors):
    return len(set(colors))


def construct():
    """
    Construct a Gallai coloring of K_16 with 20 distinct edge colors,
    then attempt local search to add more colors.
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
        return c

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

    if time.time() < deadline:
        base_colors = colors[:]
        base_distinct = _count_distinct(base_colors)
        
        if not _is_valid_gallai(base_colors):
            return base_colors
        
        improved = base_colors[:]
        best_distinct = base_distinct
        next_color = 20  # next available color index
        
        # Helper: try a set of edge recolorings and return (valid, new_distinct)
        def try_recoloring(current_colors, edge_color_map):
            """
            edge_color_map: dict from edge_idx -> new_color
            Returns (is_valid, new_distinct_count)
            """
            test = current_colors[:]
            changed = []
            for eidx, new_c in edge_color_map.items():
                if test[eidx] != new_c:
                    test[eidx] = new_c
                    changed.append(eidx)
            
            if not changed:
                return _is_valid_gallai(test), _count_distinct(test)
            
            valid = _is_valid_gallai_fast(test, changed)
            if valid:
                return True, _count_distinct(test)
            return False, 0
        
        # Phase 1: Try to split each monochromatic cut into 2 colors
        # Define all the cuts from the partition tree
        cuts = []
        
        # Node A: L0 vs L1
        cut_a_edges = []
        for u in L0:
            for v in L1:
                cut_a_edges.append(_edge_idx(u, v))
        cuts.append(cut_a_edges)
        
        # Node B: L2 vs L3
        cut_b_edges = []
        for u in L2:
            for v in L3:
                cut_b_edges.append(_edge_idx(u, v))
        cuts.append(cut_b_edges)
        
        # Node C: (L0+L1) vs (L2+L3)
        cut_c_edges = []
        for u in L0 + L1:
            for v in L2 + L3:
                cut_c_edges.append(_edge_idx(u, v))
        cuts.append(cut_c_edges)
        
        # Node D: L4 vs L5
        cut_d_edges = []
        for u in L4:
            for v in L5:
                cut_d_edges.append(_edge_idx(u, v))
        cuts.append(cut_d_edges)
        
        # Root: (L0+L1+L2+L3) vs (L4+L5)
        cut_root_edges = []
        for u in L0 + L1 + L2 + L3:
            for v in L4 + L5:
                cut_root_edges.append(_edge_idx(u, v))
        cuts.append(cut_root_edges)
        
        # For each cut, try splitting into 2 colors in different ways
        for cut_idx, cut_edges in enumerate(cuts):
            if time.time() >= deadline:
                break
            
            current_colors = improved[:]
            n_edges = len(cut_edges)
            
            # Try different split points
            for split_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
                if time.time() >= deadline:
                    break
                
                mid = int(n_edges * split_frac)
                if mid == 0 or mid == n_edges:
                    continue
                
                new_c1 = next_color
                new_c2 = next_color + 1
                next_color += 2
                
                edge_map = {}
                for i, eidx in enumerate(cut_edges):
                    if i < mid:
                        edge_map[eidx] = new_c1
                    else:
                        edge_map[eidx] = new_c2
                
                valid, new_distinct = try_recoloring(current_colors, edge_map)
                if valid and new_distinct > best_distinct:
                    best_distinct = new_distinct
                    improved = current_colors[:]
                    for eidx, new_c in edge_map.items():
                        improved[eidx] = new_c
        
        # Phase 2: Try to give individual edges in small cuts distinct colors
        # Focus on the smallest cuts first (Node D has only 3 edges)
        for cut_idx in [3]:  # Node D
            if time.time() >= deadline:
                break
            
            cut_edges = cuts[cut_idx]
            current_colors = improved[:]
            
            # Try giving each edge a unique new color
            edge_map = {}
            for i, eidx in enumerate(cut_edges):
                edge_map[eidx] = next_color + i
            next_color += len(cut_edges)
            
            valid, new_distinct = try_recoloring(current_colors, edge_map)
            if valid and new_distinct > best_distinct:
                best_distinct = new_distinct
                improved = current_colors[:]
                for eidx, new_c in edge_map.items():
                    improved[eidx] = new_c
        
        # Phase 3: Systematic single-edge recoloring search
        # Try recoloring individual edges to new colors and check validity
        if time.time() < deadline:
            for iteration in range(50):  # limit iterations
                if time.time() >= deadline:
                    break
                
                found_improvement = False
                for eidx in range(NUM_EDGES):
                    if time.time() >= deadline:
                        break
                    
                    # Try assigning this edge a new color
                    test_color = next_color
                    valid, new_distinct = try_recoloring(improved, {eidx: test_color})
                    
                    if valid and new_distinct > best_distinct:
                        best_distinct = new_distinct
                        improved[eidx] = test_color
                        next_color += 1
                        found_improvement = True
                        break  # restart search from beginning after improvement
                
                if not found_improvement:
                    break  # no more single-edge improvements possible
        
        # Phase 4: Try swapping colors between edges to enable new color assignments
        if time.time() < deadline:
            for iteration in range(20):
                if time.time() >= deadline:
                    break
                
                found_improvement = False
                # Try pairs of edges: swap their colors, then try adding a new color
                for e1 in range(NUM_EDGES):
                    if time.time() >= deadline:
                        break
                    for e2 in range(e1 + 1, NUM_EDGES):
                        if time.time() >= deadline:
                            break
                        
                        c1 = improved[e1]
                        c2 = improved[e2]
                        if c1 == c2:
                            continue
                        
                        # Try: swap colors, then give e1 a new color
                        test_map = {e1: next_color, e2: c1}
                        # e2 gets c1, e1 gets new color
                        valid, new_distinct = try_recoloring(improved, test_map)
                        
                        if valid and new_distinct > best_distinct:
                            best_distinct = new_distinct
                            improved[e1] = next_color
                            improved[e2] = c1
                            next_color += 1
                            found_improvement = True
                            break
                    
                    if found_improvement:
                        break
                
                if not found_improvement:
                    break
        
        return improved
    
    return colors
