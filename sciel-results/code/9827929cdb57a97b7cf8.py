"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    # Lexicographic order of pairs (i,j): for i=0..N-2, j=i+1..N-1
    # Index = i*(N-1) - i*(i-1)//2 + (j - i - 1)
    # Actually let's compute: number of pairs before (i,j)
    # For each row k < i, there are N-1-k pairs
    # So sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Then within row i, j ranges from i+1 to N-1, so offset is j-(i+1)
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _check_c4_free(color_subgraph_edges):
    """
    Check if the subgraph induced by edges in color_subgraph_edges (as a set of (i,j))
    contains no C4. A C4 is a 4-cycle: four distinct vertices a,b,c,d with edges
    (a,b),(b,c),(c,d),(d,a) all in the subgraph.
    
    For a set of edges, we check all 4-vertex subsets and see if they form a C4.
    """
    edge_set = set(color_subgraph_edges)
    # Build adjacency
    adj = {}
    for (u, v) in edge_set:
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all 4-vertex subsets
    vertices = sorted(set(u for u, v in edge_set) | set(v for u, v in edge_set))
    for idx_a in range(len(vertices)):
        a = vertices[idx_a]
        for idx_b in range(idx_a + 1, len(vertices)):
            b = vertices[idx_b]
            for idx_c in range(idx_b + 1, len(vertices)):
                c = vertices[idx_c]
                for idx_d in range(idx_c + 1, len(vertices)):
                    d = vertices[idx_d]
                    # Check if (a,b,c,d) forms a C4
                    # The three possible 4-cycles on {a,b,c,d}:
                    # Cycle 1: a-b-c-d-a => edges (a,b),(b,c),(c,d),(d,a)
                    # Cycle 2: a-b-d-c-a => edges (a,b),(b,d),(d,c),(c,a)
                    # Cycle 3: a-c-b-d-a => edges (a,c),(c,b),(b,d),(d,a)
                    
                    def has_edge(u, v):
                        if u > v:
                            u, v = v, u
                        return (u, v) in edge_set
                    
                    # Cycle 1: a-b, b-c, c-d, d-a
                    if has_edge(a,b) and has_edge(b,c) and has_edge(c,d) and has_edge(d,a):
                        return False
                    # Cycle 2: a-b, b-d, d-c, c-a
                    if has_edge(a,b) and has_edge(b,d) and has_edge(d,c) and has_edge(c,a):
                        return False
                    # Cycle 3: a-c, c-b, b-d, d-a
                    if has_edge(a,c) and has_edge(c,b) and has_edge(b,d) and has_edge(d,a):
                        return False
    return True

def _is_valid_coloring(colors):
    """Check if the coloring satisfies the no-C4-with-4-distinct-colors constraint."""
    # For each 4-vertex subset, check all three 4-cycles
    # A 4-cycle has four edges; if all four have distinct colors, it's invalid.
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three cycles on {a,b,c,d}:
                    # Cycle 1: (a,b),(b,c),(c,d),(d,a)
                    # Cycle 2: (a,b),(b,d),(d,c),(c,a)
                    # Cycle 3: (a,c),(c,b),(b,d),(d,a)
                    
                    def get_color(u, v):
                        if u > v:
                            u, v = v, u
                        return colors[_edge_index(u, v)]
                    
                    # Cycle 1
                    c1 = [get_color(a,b), get_color(b,c), get_color(c,d), get_color(d,a)]
                    if len(set(c1)) == 4:
                        return False
                    # Cycle 2
                    c2 = [get_color(a,b), get_color(b,d), get_color(d,c), get_color(c,a)]
                    if len(set(c2)) == 4:
                        return False
                    # Cycle 3
                    c3 = [get_color(a,c), get_color(c,b), get_color(b,d), get_color(d,a)]
                    if len(set(c3)) == 4:
                        return False
    return True

def _count_colors(colors):
    """Count the number of distinct colors used."""
    return len(set(colors))

def _greedy_set_cover_maximize_colors():
    """
    Greedy approach: try to assign as many distinct colors as possible.
    Start with all edges unassigned. Try to assign a new color to a subset of edges
    such that the subgraph formed by that color is C4-free.
    
    Actually, the constraint is: no 4-cycle has 4 distinct colors.
    This is equivalent to: for every 4-cycle, at least two of its edges share a color.
    
    Strategy: We want to maximize the number of distinct colors.
    One approach: start with 1 color for all edges (valid, 1 color).
    Then try to "split" edges into more colors while maintaining validity.
    
    Alternative: Greedy assignment. Process edges one by one. For each edge,
    try to assign it a new color (incrementing the color count) if doing so
    doesn't violate the constraint. If not, assign it an existing color.
    
    Let's try a different approach: 
    - Start with all edges color 0.
    - For each edge, try to change its color to a new unique color.
    - If the resulting coloring is still valid, keep the new color.
    - Otherwise, revert.
    
    This is O(120 * 120 * check) which might be slow. Let's optimize.
    
    Actually, let's think about it differently. The constraint is local to 4-cycles.
    When we change the color of one edge, we only need to check 4-cycles that include that edge.
    
    How many 4-cycles include a given edge (u,v)?
    We need to pick 2 more vertices from the remaining 14, and form a 4-cycle containing (u,v).
    The number of 4-cycles containing a specific edge in K_16 is C(14,2) = 91.
    But each 4-cycle has 4 edges, so each 4-cycle is counted 4 times if we count per-edge.
    
    Let's implement: for each edge, try to give it a new color. Check only affected 4-cycles.
    """
    colors = [0] * NUM_EDGES
    next_color = 1
    edges = _build_edge_list()
    
    # Precompute for each edge, which 4-cycles it belongs to
    # A 4-cycle is identified by its 4 vertices and which cycle pattern.
    # For edge (u,v), a 4-cycle containing it uses 2 other vertices w,x.
    # The cycle must include (u,v) as one of its 4 edges.
    
    # Let's precompute: for each edge index, list of (cycle_id, position) 
    # where cycle_id identifies a specific 4-cycle (by its 4 vertices and pattern)
    # and position is which of the 4 edges in that cycle this edge is.
    
    # Actually, for checking validity after changing one edge's color,
    # we need to check all 4-cycles containing that edge.
    
    # Let's precompute for each edge, the list of 4-cycles it's in.
    # A 4-cycle on vertices {a,b,c,d} has 3 possible cycles (as defined above).
    # For a given edge (u,v), it appears in a 4-cycle {a,b,c,d} if u,v are in {a,b,c,d}
    # and the specific cycle pattern includes edge (u,v).
    
    # This is getting complex. Let's just do the full check but optimized.
    # With 120 edges, and for each edge trying a new color, we do a full validity check.
    # Full check is C(16,4) = 1820 four-vertex subsets, each with 3 cycles = 5460 checks.
    # 120 edges * 5460 = 655,200 checks. Each check is O(1). This should be fast enough.
    
    # Let's try a smarter order: process edges in a random or strategic order.
    random.seed(42)
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for edge_idx in edge_order:
        # Try to assign this edge a new color
        old_color = colors[edge_idx]
        colors[edge_idx] = next_color
        if _is_valid_coloring(colors):
            next_color += 1
        else:
            colors[edge_idx] = old_color
    
    return colors, next_color

def _local_search_improve(colors):
    """
    Try to improve the coloring by swapping colors or reassigning.
    """
    best_colors = colors[:]
    best_count = _count_colors(best_colors)
    next_color = best_count  # next available new color
    
    # Try each edge, try to give it a new color
    for edge_idx in range(NUM_EDGES):
        old_color = best_colors[edge_idx]
        best_colors[edge_idx] = next_color
        if _is_valid_coloring(best_colors):
            next_color += 1
            best_count = _count_colors(best_colors)
        else:
            best_colors[edge_idx] = old_color
    
    return best_colors, best_count

def construct():
    """
    Construct a coloring of the 120 edges of K_16 with colors in 0..119
    such that no 4-cycle has 4 distinct edge colors.
    Maximize the number of distinct colors used.
    """
    start_time = time.time()
    
    # Try multiple random orders and take the best
    best_colors = None
    best_count = 0
    
    for seed in range(50):
        if time.time() - start_time > 50:
            break
        
        random.seed(seed)
        colors, count = _greedy_set_cover_maximize_colors()
        
        if count > best_count:
            best_count = count
            best_colors = colors[:]
    
    # Try to improve with local search
    if best_colors is not None:
        if time.time() - start_time < 55:
            improved_colors, improved_count = _local_search_improve(best_colors)
            if improved_count > best_count:
                best_colors = improved_colors
                best_count = improved_count
    
    if best_colors is None:
        # Fallback: all zeros
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors

# EVOLVE-BLOCK-END
