import time

N = 40
DEADLINE_SECONDS = 85

def construct():
    t0 = time.time()
    
    # Use a different construction: two paths with shared endpoints
    # Path A: 0-1-2-...-19
    # Path B: 0-20-21-...-39
    # Connect 19-20 to form a 40-cycle
    # Add chords from 0 to create smaller cycles
    
    edges = []
    
    # Path A edges
    for i in range(19):
        edges.append((i, i + 1))
    
    # Path B edges  
    for i in range(20, 39):
        edges.append((i, i + 1))
    
    # Connect paths: 19-20 and 0 is shared
    edges.append((19, 20))
    
    # Now we have a 40-cycle: 0-1-2-...-19-20-21-...-39-0? No, 39 not connected to 0
    # Need edge (39, 0)
    edges.append((39, 0))
    
    # Now we have a 40-cycle: 0-1-...-19-20-...-39-0
    # For smaller cycles, add chords from 0
    
    # To get cycle of length L, we need a path of length L-1 from 0 back to 0
    # With the 40-cycle, we can get any cycle by taking a contiguous segment + chord
    
    # Actually, let's use a simpler approach: keep the original but try to minimize
    
    # Reset to original construction
    rim_edges = [(i, i + 1) for i in range(N - 1)]
    rim_edges.append((N - 1, 0))
    
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
            m = (L - 1) // 2
            cyc = list(range(0, 2 * m + 1))
        else:
            m = L // 2
            if m == 1:
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
