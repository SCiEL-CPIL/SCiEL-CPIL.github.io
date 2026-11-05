<<<<<<< SEARCH
    # Try different LUW parameters
    best_n = None
    best_edges = None
    best_adj = None

    # For k=5, g=9: try various (q, d) combinations
    # d should be at least ceil((g-1)/2) for the construction to potentially have girth >= g
    # Actually the LUW construction's girth depends on the specific rule set
    candidate_params = []
    for q in [k]:
        for d in range(4, 10):
            candidate_params.append((q, d))

    for q, d in candidate_params:
        if time.time() > deadline - 30:
            break
        try:
            n, edges, order, index = luw_construction(q, d)
            adj = build_adj(n, edges)
            # Verify
            if all(len(adj[i]) == k for i in range(n)):
                if check_girth_nonbacktracking(n, adj, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj
        except Exception:
            continue

    # If no valid LUW graph found, use a larger d
    if best_n is None:
        for d in range(10, 15):
            if time.time() > deadline - 20:
                break
            try:
                n, edges, order, index = luw_construction(k, d)
                adj = build_adj(n, edges)
                if all(len(adj[i]) == k for i in range(n)):
                    if check_girth_nonbacktracking(n, adj, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                            best_adj = adj
                        break
            except Exception:
                continue

    if best_n is None:
        # Last resort: use a very large d
        n, edges, order, index = luw_construction(k, 12)
        best_n = n
        best_edges = edges
        best_adj = build_adj(n, edges)
=======
    # Try different LUW parameters
    best_n = None
    best_edges = None
    best_adj = None

    # For k=5, g=9: try various (q, d) combinations
    # Start with smaller d values that may yield smaller graphs
    candidate_params = []
    for q in [k]:
        for d in range(3, 12):
            candidate_params.append((q, d))

    for q, d in candidate_params:
        if time.time() > deadline - 40:
            break
        try:
            n, edges, order, index = luw_construction(q, d)
            adj = build_adj(n, edges)
            # Verify
            if all(len(adj[i]) == k for i in range(n)):
                if check_girth_nonbacktracking(n, adj, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj
        except Exception:
            continue

    # If no valid LUW graph found, use a larger d
    if best_n is None:
        for d in range(12, 16):
            if time.time() > deadline - 25:
                break
            try:
                n, edges, order, index = luw_construction(k, d)
                adj = build_adj(n, edges)
                if all(len(adj[i]) == k for i in range(n)):
                    if check_girth_nonbacktracking(n, adj, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                            best_adj = adj
                        break
            except Exception:
                continue

    if best_n is None:
        # Last resort: use a very large d
        n, edges, order, index = luw_construction(k, 14)
        best_n = n
        best_edges = edges
        best_adj = build_adj(n, edges)
>>>>>>> REPLACE
