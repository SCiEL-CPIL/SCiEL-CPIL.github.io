<<<<<<< SEARCH
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s limit
    
    best_n = None
    best_edges = None
    
    # For the specific case k=4, g=10, use the known cage construction
    if k == 4 and g == 10:
        # Try the known 320-vertex construction
        candidates = [
            _construct_cage_4_10,
            _construct_cage_4_10_v2,
        ]
        
        for construct_fn in candidates:
            if time.time() > deadline:
                break
            try:
                n, edges = construct_fn()
                # Verify the graph is valid
                if _check_girth_nonbacktracking(n, edges, g):
                    # Check degree
                    deg = [0] * n
                    for u, v in edges:
                        deg[u] += 1
                        deg[v] += 1
                    if all(d == k for d in deg):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                            if n == 320:  # Known cage size
                                break
            except Exception:
                pass
    
    # Try LUW constructions as fallback
    candidates_d = []
    for d in range(3, 10):
        if 2 * d + 2 >= g:
            candidates_d.append(d)
    
    candidates_d.sort()
    
    for d in candidates_d:
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(d, k)
            if _check_girth_nonbacktracking(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
    
    # If no valid graph found, use the smallest d that satisfies girth condition
    if best_n is None:
        d = max(3, (g - 2) // 2)
        try:
            n, edges = _luw_construct(d, k)
            best_n = n
            best_edges = edges
        except Exception:
            # Fallback: create a trivial graph
            best_n = k + 1
            best_edges = []
            for i in range(k):
                best_edges.append((0, i + 1))
    
    return best_n, best_edges
=======
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    best_n = None
    best_edges = None
    
    # For the specific case k=4, g=10, use the known cage construction
    if k == 4 and g == 10:
        # Try the 320-vertex constructions
        candidates = [
            _construct_cage_4_10,
            _construct_cage_4_10_v2,
        ]
        
        for construct_fn in candidates:
            if time.time() > deadline:
                break
            try:
                n, edges = construct_fn()
                # Verify the graph is valid
                if _check_girth_nonbacktracking(n, edges, g):
                    # Check degree
                    deg = [0] * n
                    for u, v in edges:
                        deg[u] += 1
                        deg[v] += 1
                    if all(d == k for d in deg):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                            if n == 320:  # Known cage size
                                break
            except Exception:
                pass
    
    # Try LUW constructions as fallback
    candidates_d = []
    for d in range(3, 12):
        if 2 * d + 2 >= g:
            candidates_d.append(d)
    
    candidates_d.sort()
    
    for d in candidates_d:
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(d, k)
            if _check_girth_nonbacktracking(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
    
    # If no valid graph found, use the smallest d that satisfies girth condition
    if best_n is None:
        d = max(3, (g - 2) // 2)
        try:
            n, edges = _luw_construct(d, k)
            best_n = n
            best_edges = edges
        except Exception:
            # Fallback: create a trivial graph
            best_n = k + 1
            best_edges = []
            for i in range(k):
                best_edges.append((0, i + 1))
    
    return best_n, best_edges
>>>>>>> REPLACE
