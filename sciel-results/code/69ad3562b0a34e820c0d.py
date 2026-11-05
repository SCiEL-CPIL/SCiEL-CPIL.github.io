import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + j - i - 1

def _build_edge_list():
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _build_cycles():
    """Return list of 4-cycles, each as a tuple of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}
                    # Cycle 1: a-b, b-c, c-d, d-a
                    e1 = (_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(a, d))
                    # Cycle 2: a-b, b-d, d-c, c-a
                    e2 = (_edge_idx(a, b), _edge_idx(b, d), _edge_idx(c, d), _edge_idx(a, c))
                    # Cycle 3: a-c, c-b, b-d, d-a
                    e3 = (_edge_idx(a, c), _edge_idx(b, c), _edge_idx(b, d), _edge_idx(a, d))
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

def _check_cycle_valid(cycle, colors):
    """Check if a 4-cycle has fewer than 4 distinct colors."""
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
        return True
    return False

def _count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    edges = _build_edge_list()
    cycles = _build_cycles()
    
    # Build edge-to-cycles mapping for efficient checking
    edge_to_cycles = {i: [] for i in range(NUM_EDGES)}
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    # Initial construction: use a structured approach
    # Partition into 4 groups of 4
    # Within each group, use a small set of colors
    # Between groups, use colors based on group pair
    colors = [0] * NUM_EDGES
    
    # Group assignment: vertex v in group v % 4
    def group(v):
        return v // 4
    
    # Within-group colors: use colors 0-3 for each group
    # Between-group colors: use colors 4-15 for each pair of groups
    group_pair_color = {}
    color_id = 4
    for g1 in range(4):
        for g2 in range(g1 + 1, 4):
            group_pair_color[(g1, g2)] = color_id
            color_id += 1
    
    for idx, (i, j) in enumerate(edges):
        gi, gj = group(i), group(j)
        if gi == gj:
            # Within group: use color based on position
            colors[idx] = (i + j) % 4
        else:
            key = (min(gi, gj), max(gi, gj))
            colors[idx] = group_pair_color[key]
    
    # Verify initial construction
    valid = all(_check_cycle_valid(c, colors) for c in cycles)
    if not valid:
        # Fallback: all same color
        colors = [0] * NUM_EDGES
    
    best_colors = colors[:]
    best_count = _count_distinct_colors(colors)
    
    # Local search with simulated annealing
    def is_valid(colors):
        for c in cycles:
            if not _check_cycle_valid(c, colors):
                return False
        return True
    
    def is_valid_after_change(colors, edge_idx, new_color):
        """Check validity after changing one edge."""
        for ci in edge_to_cycles[edge_idx]:
            cycle = cycles[ci]
            c0 = colors[cycle[0]]
            c1 = colors[cycle[1]]
            c2 = colors[cycle[2]]
            c3 = colors[cycle[3]]
            if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
                continue
            return False
        return True
    
    # Simulated annealing
    temp = 5.0
    cool_rate = 0.999
    
    current = colors[:]
    current_count = _count_distinct_colors(current)
    
    while time.time() < deadline:
        if temp < 0.1:
            break
        
        # Pick a random edge
        ei = random.randint(0, NUM_EDGES - 1)
        old_color = current[ei]
        
        # Try a new color
        if random.random() < 0.5:
            new_color = random.randint(0, MAX_COLOR)
        else:
            new_color = random.randint(0, min(30, MAX_COLOR))
        
        if new_color == old_color:
            continue
        
        # Check if change is valid
        if not is_valid_after_change(current, ei, new_color):
            continue
        
        current[ei] = new_color
        new_count = _count_distinct_colors(current)
        
        delta = new_count - current_count
        
        if delta > 0 or random.random() < (2.718281828 ** (delta / temp)):
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_colors = current[:]
        
        temp *= cool_rate
    
    # Return best found
    result = best_colors[:]
    # Ensure all colors are in 0..119
    for i in range(len(result)):
        if result[i] < 0 or result[i] > MAX_COLOR:
            result[i] = 0
    
    return result
