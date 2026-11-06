# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Returns a list `cycles_for_edge` of length NUM_EDGES.
    cycles_for_edge[e] is a list of tuples (e1, e2, e3) where 
    {e, e1, e2, e3} is a 4-cycle.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    cycles_for_edge = [[] for _ in range(NUM_EDGES)]
    
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    def get_idx(u, v):
                        if u > v: u, v = v, u
                        return edge_to_idx[(u, v)]
                    
                    # Cycle 1: a-b-c-d-a
                    e1, e2, e3, e4 = get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)
                    cyc = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc):
                        others = [cyc[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 2: a-b-d-c-a
                    e1, e2, e3, e4 = get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)
                    cyc = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc):
                        others = [cyc[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 3: a-c-b-d-a
                    e1, e2, e3, e4 = get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)
                    cyc = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc):
                        others = [cyc[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                        
    return cycles_for_edge

CYCLES_FOR_EDGE = _precompute_cycles()

def _greedy_smart(seed):
    random.seed(seed)
    colors = [-1] * NUM_EDGES
    next_color = 0
    uncolored = set(range(NUM_EDGES))
    
    threats = [0] * NUM_EDGES
    
    while uncolored:
        candidates = [e for e in uncolored if threats[e] == 0]
        
        if not candidates:
            break
            
        if len(candidates) == 1:
            e_star = candidates[0]
        else:
            # Shuffle candidates to randomize tie-breaking
            random.shuffle(candidates)
            
            best_cost = float('inf')
            e_star = candidates[0]
            
            for e in candidates:
                cost = 0
                for others in CYCLES_FOR_EDGE[e]:
                    o1, o2, o3 = others
                    
                    if o1 in uncolored:
                        if colors[o2] != -1 and colors[o3] != -1 and colors[o2] != colors[o3]:
                            cost += 1
                    if o2 in uncolored:
                        if colors[o1] != -1 and colors[o3] != -1 and colors[o1] != colors[o3]:
                            cost += 1
                    if o3 in uncolored:
                        if colors[o1] != -1 and colors[o2] != -1 and colors[o1] != colors[o2]:
                            cost += 1
                            
                if cost < best_cost:
                    best_cost = cost
                    e_star = e
                    if cost == 0:
                        break
                        
        colors[e_star] = next_color
        next_color += 1
        uncolored.remove(e_star)
        
        for others in CYCLES_FOR_EDGE[e_star]:
            o1, o2, o3 = others
            
            if o1 in uncolored:
                if colors[o2] != -1 and colors[o3] != -1 and colors[o2] != colors[o3]:
                    threats[o1] += 1
            if o2 in uncolored:
                if colors[o1] != -1 and colors[o3] != -1 and colors[o1] != colors[o3]:
                    threats[o2] += 1
            if o3 in uncolored:
                if colors[o1] != -1 and colors[o2] != -1 and colors[o1] != colors[o2]:
                    threats[o3] += 1
                    
    for e in uncolored:
        colors[e] = 0
        
    return colors

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    for seed in range(200):
        if time.time() - start_time > 55:
            break
        
        colors = _greedy_smart(seed)
        count = _count_colors(colors)
        if count > best_count:
            best_count = count
            best_colors = colors[:]
            
    return best_colors

# EVOLVE-BLOCK-END
