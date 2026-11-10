N = 16
NUM_EDGES = N * (N - 1) // 2

def _build_cycles():
    """Build all 4-cycles: each is a set of 4 edge indices."""
    cycles = []
    # A 4-cycle is determined by 4 vertices a<b<c<d, and the 3 possible cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d)
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on vertices {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    edges1 = {(a,b), (b,c), (c,d), (d,a)}
                    # Cycle 2: a-b-d-c-a
                    edges2 = {(a,b), (b,d), (d,c), (c,a)}
                    # Cycle 3: a-c-b-d-a
                    edges3 = {(a,c), (c,b), (b,d), (d,a)}
                    for edge_set in [edges1, edges2, edges3]:
                        cycles.append(edge_set)
    return cycles

def _edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + j - i - 1

def construct():
    import time
    import random
    
    deadline = time.time() + 55  # 55 second safety margin
    
    # Build edge list and index mapping
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_index(i, j)
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
    
    # Build all 4-cycles as lists of edge indices
    raw_cycles = _build_cycles()
    cycles = []
    for edge_set in raw_cycles:
        cycle_edges = []
        for (u, v) in edge_set:
            if u > v:
                u, v = v, u
            cycle_edges.append(edge_to_idx[(u, v)])
        cycles.append(cycle_edges)
    
    # For each edge, find which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    def is_valid(coloring):
        """Check if coloring has no rainbow 4-cycle."""
        for cycle in cycles:
            colors = set(coloring[e] for e in cycle)
            if len(colors) == 4:
                return False
        return True
    
    def check_edge_assignment(coloring, edge_idx, new_color):
        """Check if assigning new_color to edge_idx maintains validity."""
        for ci in edge_to_cycles[edge_idx]:
            cycle = cycles[ci]
            other_colors = set()
            for e in cycle:
                if e != edge_idx:
                    other_colors.add(coloring[e])
            if len(other_colors) == 3 and new_color not in other_colors:
                return False
        return True
    
    # Initial coloring: use a structured pattern to maximize colors
    # Color based on (i+j) mod something, or other patterns
    random.seed(42)
    
    # Try several initial colorings and pick the best
    best_result = [0] * NUM_EDGES
    best_count = 1
    
    # Strategy 1: color = (i + j) % 16
    coloring1 = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edges):
        coloring1[idx] = (i + j) % 16
    if is_valid(coloring1):
        c = len(set(coloring1))
        if c > best_count:
            best_count = c
            best_result = coloring1[:]
    
    # Strategy 2: color = (i * 2 + j) % 32
    coloring2 = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edges):
        coloring2[idx] = (i * 2 + j) % 32
    if is_valid(coloring2):
        c = len(set(coloring2))
        if c > best_count:
            best_count = c
            best_result = coloring2[:]
    
    # Strategy 3: color = (i * 5 + j * 3) % 64
    coloring3 = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edges):
        coloring3[idx] = (i * 5 + j * 3) % 64
    if is_valid(coloring3):
        c = len(set(coloring3))
        if c > best_count:
            best_count = c
            best_result = coloring3[:]
    
    # Strategy 4: color = (i ^ j) for first 16, then (i ^ j) + 16 for rest
    coloring4 = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edges):
        coloring4[idx] = (i ^ j) % 120
    if is_valid(coloring4):
        c = len(set(coloring4))
        if c > best_count:
            best_count = c
            best_result = coloring4[:]
    
    # Greedy improvement: try to introduce new colors
    current = best_result[:]
    current_count = len(set(current))
    
    # Get list of colors in use
    def get_used_colors(col):
        return set(col)
    
    # Try to assign new colors to edges
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        improved = False
        used = get_used_colors(current)
        
        # Try edges in random order
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline - 5:
                break
            
            # Try to assign a new color to this edge
            # Pick a random unused color
            all_colors = set(range(120))
            available = list(all_colors - used)
            if not available:
                break
            
            random.shuffle(available)
            
            for new_color in available[:10]:  # Try up to 10 candidates
                if check_edge_assignment(current, e, new_color):
                    current[e] = new_color
                    used.add(new_color)
                    current_count = len(used)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try swap moves to enable new colors
            swap_improved = False
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                
                e1 = random.randint(0, NUM_EDGES - 1)
                e2 = random.randint(0, NUM_EDGES - 1)
                if e1 == e2:
                    continue
                
                old1, old2 = current[e1], current[e2]
                if old1 == old2:
                    continue
                
                # Swap
                current[e1], current[e2] = old2, old1
                
                # Check if still valid
                valid = True
                for ci in edge_to_cycles[e1]:
                    cycle = cycles[ci]
                    if len(set(current[e] for e in cycle)) == 4:
                        valid = False
                        break
                if valid:
                    for ci in edge_to_cycles[e2]:
                        cycle = cycles[ci]
                        if len(set(current[e] for e in cycle)) == 4:
                            valid = False
                            break
                
                if valid:
                    # Try to add a new color
                    used = get_used_colors(current)
                    all_colors = set(range(120))
                    available = list(all_colors - used)
                    if available:
                        random.shuffle(available)
                        for new_color in available[:5]:
                            if check_edge_assignment(current, e1, new_color):
                                current[e1] = new_color
                                current_count = len(get_used_colors(current))
                                swap_improved = True
                                break
                            elif check_edge_assignment(current, e2, new_color):
                                current[e2] = new_color
                                current_count = len(get_used_colors(current))
                                swap_improved = True
                                break
                
                # Undo swap
                current[e1], current[e2] = old1, old2
                
                if swap_improved:
                    break
            
            if not swap_improved and not improved:
                break
    
    if current_count > best_count:
        best_count = current_count
        best_result = current[:]
    
    # Final validation
    if not is_valid(best_result):
        # Fallback to all zeros
        best_result = [0] * NUM_EDGES
    
    return best_result
