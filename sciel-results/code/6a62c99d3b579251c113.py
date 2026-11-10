import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def get_all_4_cycles():
    """Generate all (vertex_set, cycle_edges) for all 4-vertex subsets and their 3 cycles."""
    cycles = []
    for verts in itertools.combinations(range(N), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        edges1 = [(a,b), (b,c), (c,d), (d,a)]
        # Cycle 2: a-b-d-c-a
        edges2 = [(a,b), (b,d), (d,c), (c,a)]
        # Cycle 3: a-c-b-d-a
        edges3 = [(a,c), (c,b), (b,d), (d,a)]
        for edges in [edges1, edges2, edges3]:
            idxs = [edge_index(u, v) for u, v in edges]
            cycles.append(tuple(idxs))
    return cycles

ALL_CYCLES = get_all_4_cycles()

def count_distinct_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    for cycle in ALL_CYCLES:
        colors = set(coloring[i] for i in cycle)
        if len(colors) > 3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline, 5 sec safety margin
    
    # Initial coloring: use a structured pattern
    # Assign each vertex a 4-bit value, color edge by XOR of vertex values mapped to color
    # Use a hash-like function to spread colors but maintain structure
    
    random.seed(42)
    
    # Start with a coloring based on vertex partition
    # Partition into 4 groups of 4
    group_of = [i // 4 for i in range(N)]
    
    # Initial: color based on group pair
    coloring = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = edge_index(i, j)
            gi, gj = group_of[i], group_of[j]
            # Assign color based on group pair
            if gi == gj:
                coloring[idx] = gi * 30 + (i % 4) + (j % 4)
            else:
                coloring[idx] = gi * 30 + gj * 3 + (i % 4) * 2 + (j % 4)
    
    # Ensure colors are in range
    coloring = [c % MAX_COLORS for c in coloring]
    
    best_coloring = list(coloring)
    best_count = count_distinct_colors(coloring)
    
    # Check validity of initial
    if is_valid(coloring):
        pass
    else:
        # Fallback: use a simple valid coloring with few colors
        # Color all edges with 3 colors based on vertex index mod 3
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_index(i, j)
                coloring[idx] = (i + j) % 3
        best_coloring = list(coloring)
        best_count = count_distinct_colors(coloring)
    
    # Local search: try to change individual edge colors to increase distinct count
    # while maintaining validity
    
    # Precompute for each edge, which cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(ALL_CYCLES):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    while time.time() < deadline:
        improved = False
        # Try random edge changes
        for _ in range(2000):
            if time.time() > deadline:
                break
            ei = random.randint(0, NUM_EDGES - 1)
            old_color = coloring[ei]
            
            # Try a new random color
            new_color = random.randint(0, MAX_COLORS - 1)
            if new_color == old_color:
                continue
            
            coloring[ei] = new_color
            
            # Check validity only for cycles containing this edge
            valid = True
            for ci in edge_to_cycles[ei]:
                cycle = ALL_CYCLES[ci]
                colors = set(coloring[i] for i in cycle)
                if len(colors) > 3:
                    valid = False
                    break
            
            if valid:
                new_count = count_distinct_colors(coloring)
                if new_count > best_count:
                    best_count = new_count
                    best_coloring = list(coloring)
                    improved = True
                else:
                    # Revert if not improving
                    coloring[ei] = old_color
            else:
                coloring[ei] = old_color
        
        if not improved and time.time() > deadline - 10:
            break
    
    return best_coloring
