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
    
    edge_set = set()
    for a, b in edges:
        edge_set.add((min(a, b), max(a, b)))
        
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L % 2 == 1:
            # L = 2m + 1 => m = (L-1)//2
            m = (L - 1) // 2
            # Cycle: 0, 1, ..., 2m
            cyc = list(range(0, 2 * m + 1))
        else:
            # L = 2m => m = L//2
            m = L // 2
            # Cycle: 0, 2, 3, ..., 2m
            # Vertices: 0, then 2 to 2m
            if m == 1: # L=2 not possible
                continue
            cyc = [0] + list(range(2, 2 * m + 1))
            
        cycles.append(cyc)
        
    # Validation
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
            
    if not ok:
        # Fallback to seed if something goes wrong
        path_edges = [(i, i + 1) for i in range(N - 1)]
        spoke_edges = [(i, 0) for i in range(2, N)]
        edges = path_edges + spoke_edges
        seen = set()
        unique_edges = []
        for a, b in edges:
            if a == b: continue
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_edges.append((a, b))
        edges = unique_edges
        cycles = [list(range(k)) for k in range(3, N + 1)]

    return {'edges': edges, 'cycles': cycles}
