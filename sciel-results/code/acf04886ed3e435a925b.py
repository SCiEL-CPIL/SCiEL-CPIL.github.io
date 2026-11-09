import time

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    # Rim edges
    rim_edges = [(i, i + 1) for i in range(N - 1)]
    rim_edges.append((N - 1, 0))
    
    # Chord edges: (0, 2k) for k=1..19
    chord_edges = [(0, 2 * k) for k in range(1, N // 2)]
    
    edges = rim_edges + chord_edges
    
    # De-duplicate
    seen = set()
    unique_edges = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append((a, b))
    edges = unique_edges
    
    # Define cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L % 2 == 1:
            m = (L - 1) // 2
            cyc = list(range(0, 2 * m + 1))
        else:
            m = L // 2
            if m == 1:
                continue
            cyc = [0] + list(range(2, 2 * m + 1))
        cycles.append(cyc)
    
    # Collect all edges needed by the cycles
    needed_edges = set()
    for cyc in cycles:
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            needed_edges.add((min(u, v), max(u, v)))
    
    # Keep only edges that are needed by at least one cycle
    filtered_edges = []
    for a, b in edges:
        key = (min(a, b), max(a, b))
        if key in needed_edges:
            filtered_edges.append((a, b))
    
    # Validate
    edge_set = set()
    for a, b in filtered_edges:
        edge_set.add((min(a, b), max(a, b)))
    
    ok = True
    for cyc in cycles:
        if len(set(cyc)) != len(cyc):
            ok = False
            break
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if (min(u, v), max(u, v)) not in edge_set:
                ok = False
                break
        if not ok:
            break
    
    if ok:
        edges = filtered_edges
    else:
        # Fallback: use all original edges
        edge_set = set()
        for a, b in edges:
            edge_set.add((min(a, b), max(a, b)))
    
    return {'edges': edges, 'cycles': cycles}
