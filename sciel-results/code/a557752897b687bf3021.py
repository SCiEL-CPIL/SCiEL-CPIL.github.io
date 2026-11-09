"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16

def _edge_idx(i, j):
    """Return the index for edge (i,j) in the lexicographic ordering of pairs."""
    # Pairs are ordered lexicographically: (0,1), (0,2), ..., (0,15), (1,2), ...
    # Index = sum_{k=0}^{i-1} (N - 1 - k) + (j - i - 1)
    #       = i*(N-1) - i*(i-1)//2 + j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_map():
    """Build a dictionary mapping (i,j) to index for quick lookup."""
    emap = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            emap[(i, j)] = idx
            idx += 1
    return emap

_EDGE_MAP = _build_edge_map()

def _get_edge_color(colors, i, j):
    return colors[_EDGE_MAP[(i, j)]]

def _check_four_cycle(colors, a, b, c, d):
    """
    Check if the four-cycle a-b-c-d-a has four distinct colors.
    The four edges of this cycle are: (a,b), (b,c), (c,d), (d,a)
    """
    c1 = _get_edge_color(colors, a, b)
    c2 = _get_edge_color(colors, b, c)
    c3 = _get_edge_color(colors, c, d)
    c4 = _get_edge_color(colors, d, a)
    return len({c1, c2, c3, c4}) == 4

def _all_four_cycles():
    """
    Generate all four-cycles in K_16. A four-cycle is determined by 4 distinct vertices.
    For vertices {a,b,c,d}, there are 3 distinct cycles:
      1. a-b-c-d-a (edges: ab, bc, cd, da)
      2. a-b-d-c-a (edges: ab, bd, dc, ca)
      3. a-c-b-d-a (edges: ac, cb, bd, da)
    We return all such cycles as tuples of 4 vertex indices.
    """
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # The three distinct 4-cycles on these 4 vertices:
        cycles.append((a, b, c, d))  # a-b-c-d-a
        cycles.append((a, b, d, c))  # a-b-d-c-a
        cycles.append((a, c, b, d))  # a-c-b-d-a
    return cycles

_ALL_CYCLES = _all_four_cycles()

def _count_violations(colors):
    """Count the number of four-cycles with four distinct edge colors."""
    count = 0
    for a, b, c, d in _ALL_CYCLES:
        c1 = colors[_EDGE_MAP[(a, b)]]
        c2 = colors[_EDGE_MAP[(b, c)]]
        c3 = colors[_EDGE_MAP[(c, d)]]
        c4 = colors[_EDGE_MAP[(d, a)]]
        if len({c1, c2, c3, c4}) == 4:
            count += 1
    return count

def _count_colors(colors):
    return len(set(colors))

def _greedy_coloring(seed=0):
    """
    Greedy approach: try to assign colors to maximize distinct colors while
    avoiding four distinct colors on any 4-cycle.
    
    Strategy: Use a recursive Gallai-style partitioning approach.
    Partition vertices into 2 groups. Within each group, use a valid coloring
    (recursively or simple). Between groups, use a single color.
    This ensures no 4-cycle has 4 distinct colors because any 4-cycle either
    has all edges within one part, all within the other, or crosses between
    parts (in which case at least two edges share the cross color).
    
    Actually, let's think more carefully. A 4-cycle a-b-c-d-a where a,c are in
    part 1 and b,d are in part 2 has edges: a-b (cross), b-c (cross), c-d (cross),
    d-a (cross). All four are cross edges, so they all have the same color. Valid.
    
    A 4-cycle where a,b in part 1 and c,d in part 2: edges a-b (in1), b-c (cross),
    c-d (in2), d-a (cross). Colors: c_in1, cross, c_in2, cross. At most 3 distinct. Valid.
    
    A 4-cycle where all 4 in same part: handled recursively.
    
    So this partitioning works! We can use different colors in the two subparts
    to maximize color count.
    """
    rng = random.Random(seed)
    
    def solve(vertices):
        """Return a list of colors for edges among vertices (in lexicographic order of those vertices)."""
        if len(vertices) <= 3:
            # No 4-cycles possible, use all distinct colors
            n_edges = len(vertices) * (len(vertices) - 1) // 2
            # Use unique color offsets
            base = rng.randint(0, 50)
            return [(base + i) % 120 for i in range(n_edges)]
        
        # Split vertices into two groups
        mid = len(vertices) // 2
        part1 = vertices[:mid]
        part2 = vertices[mid:]
        
        # Recursively solve each part
        colors1 = solve(part1)
        colors2 = solve(part2)
        
        # Cross edges get a single color
        cross_color = rng.randint(0, 119)
        
        # Build the full edge list
        # Map from local vertex indices to global
        # vertices is a list of global vertex indices
        n1 = len(part1)
        n2 = len(part2)
        
        # Total edges among all vertices
        total = len(vertices)
        n_total_edges = total * (total - 1) // 2
        
        # Build mapping: for each edge in the full vertex set, determine its color
        # The edges are ordered lexicographically on the local indices 0..total-1
        edge_colors = [0] * n_total_edges
        idx = 0
        for i in range(total):
            for j in range(i + 1, total):
                if i < n1 and j < n1:
                    # Edge in part1
                    local_i = i
                    local_j = j
                    edge_colors[idx] = colors1[local_i * (n1 - 1) - local_i * (local_i - 1) // 2 + (local_j - local_i - 1)]
                elif i >= n1 and j >= n1:
                    # Edge in part2
                    local_i = i - n1
                    local_j = j - n1
                    edge_colors[idx] = colors2[local_i * (n2 - 1) - local_i * (local_i - 1) // 2 + (local_j - local_i - 1)]
                else:
                    # Cross edge
                    edge_colors[idx] = cross_color
                idx += 1
        
        return edge_colors
    
    # Solve for all 16 vertices
    vertices = list(range(N))
    return solve(vertices)

def _random_search(seed=0, time_limit=55.0):
    """
    Randomized search: start with a valid coloring, then try to modify it
    to increase the number of distinct colors.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Start with a valid partition-based coloring
    colors = _greedy_coloring(seed)
    best_colors = list(colors)
    best_count = _count_colors(colors)
    assert _count_violations(colors) == 0, "Initial coloring has violations"
    
    # Try local search: pick a random edge, change its color, check validity
    all_edges = list(range(N * (N - 1) // 2))
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to modify
        edge = rng.choice(all_edges)
        old_color = colors[edge]
        
        # Try a new color
        new_color = rng.randint(0, 119)
        if new_color == old_color:
            continue
        
        colors[edge] = new_color
        
        # Check if still valid
        if _count_violations(colors) == 0:
            new_count = _count_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = list(colors)
        else:
            colors[edge] = old_color
    
    return best_colors

def _construct_gallai():
    """
    Construct a valid coloring using Gallai partitioning with maximum color diversity.
    
    For n=16, split into 8+8. Each 8-split into 4+4. Each 4-split into 2+2.
    Each 2 has 1 edge.
    
    This gives a hierarchical structure. We can assign different cross colors at
    each level to maximize total distinct colors.
    """
    # For a 2-vertex component: 1 edge
    # For a 4-vertex component (2+2): 1+1+4 = 6 edges (2 within pairs, 4 cross)
    #   Cross edges all same color, so 2 distinct colors max
    # For an 8-vertex component (4+4): 6+6+16 = 28 edges
    #   16 cross edges all same color, so at most 2+2+1 = 5 distinct colors
    # For a 16-vertex component (8+8): 28+28+64 = 120 edges
    #   64 cross edges all same color, so at most 5+5+1 = 11 distinct colors
    
    # To maximize, we want different cross colors at each level and different
    # internal colorings.
    
    # Let's build this explicitly with a recursive function that assigns
    # unique color ranges at each level.
    
    def build(vertices, color_offset, color_span):
        """
        Build coloring for vertices. Returns list of edge colors.
        color_offset: starting color
        color_span: number of colors available for this subproblem
        """
        n = len(vertices)
        if n <= 3:
            # Use all distinct colors
            n_edges = n * (n - 1) // 2
            return [(color_offset + i) % 120 for i in range(n_edges)]
        
        mid = n // 2
        part1 = vertices[:mid]
        part2 = vertices[mid:]
        n1 = len(part1)
        n2 = len(part2)
        
        # Allocate colors: half for part1, half for part2, one for cross
        span1 = color_span // 3
        span2 = color_span // 3
        cross_color = (color_offset + color_span - 1) % 120
        
        colors1 = build(part1, color_offset, span1)
        colors2 = build(part2, color_offset + span1, span2)
        
        # Build full edge list
        n_total_edges = n * (n - 1) // 2
        edge_colors = [0] * n_total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i < mid and j < mid:
                    local_i = i
                    local_j = j
                    edge_colors[idx] = colors1[local_i * (mid - 1) - local_i * (local_i - 1) // 2 + (local_j - local_i - 1)]
                elif i >= mid and j >= mid:
                    local_i = i - mid
                    local_j = j - mid
                    edge_colors[idx] = colors2[local_i * (n2 - 1) - local_i * (local_i - 1) // 2 + (local_j - local_i - 1)]
                else:
                    edge_colors[idx] = cross_color
                idx += 1
        
        return edge_colors
    
    vertices = list(range(N))
    return build(vertices, 0, 120)

def construct():
    """Return 120 integer colors in 0..119 for lexicographically ordered pairs."""
    start_time = time.time()
    
    # Start with the Gallai partition coloring
    colors = _construct_gallai()
    best_colors = list(colors)
    best_count = _count_colors(colors)
    
    assert _count_violations(colors) == 0, "Gallai coloring has violations"
    
    # Try to improve with local search
    rng = random.Random(42)
    all_edges = list(range(N * (N - 1) // 2))
    
    time_limit = 50.0
    while time.time() - start_time < time_limit:
        # Pick a random edge
        edge = rng.choice(all_edges)
        old_color = colors[edge]
        new_color = rng.randint(0, 119)
        if new_color == old_color:
            continue
        
        colors[edge] = new_color
        
        if _count_violations(colors) == 0:
            new_count = _count_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = list(colors)
        else:
            colors[edge] = old_color
    
    return best_colors
