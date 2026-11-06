<<<<<<< SEARCH
    # ---------- Search from high n down ----------
    best = None
    for n in range(43, 29, -1):
        if time.time() > DEADLINE - 5:
            break
        seeds = 0
        max_seeds = 3000
        while time.time() < DEADLINE - 5 and seeds < max_seeds:
            adj_mask = generate_greedy_c4_free(n, seeds)

            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                seeds += 1
                continue

            # Check independence number
            if not has_independent_set(adj_mask, n, 11):
                edges = build_edges(n, adj_mask)
                best = {'n': n, 'edges': edges}
                break

            seeds += 1

        if best is not None:
            break

    if best is None:
        best = fallback()

    return best
=======
    # ---------- Search from high n down ----------
    best = None
    for n in range(43, 29, -1):
        if time.time() > DEADLINE - 5:
            break
        seeds = 0
        max_seeds = 2000
        while time.time() < DEADLINE - 5 and seeds < max_seeds:
            # Alternate between two greedy strategies
            if seeds % 2 == 0:
                adj_mask = generate_greedy_c4_free(n, seeds)
            else:
                adj_mask = generate_greedy_biased(n, seeds)

            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                seeds += 1
                continue

            # Try local search refinement
            if time.time() < DEADLINE - 15:
                adj_mask = local_search_refine(n, adj_mask, seeds + 100000, time_budget=3.0)
                # Re-verify C4-free after local search
                if not is_c4_free(adj_mask, n):
                    seeds += 1
                    continue

            # Check independence number
            if not has_independent_set(adj_mask, n, 11):
                edges = build_edges(n, adj_mask)
                best = {'n': n, 'edges': edges}
                break

            seeds += 1

        if best is not None:
            break

    if best is None:
        best = fallback()

    return best
>>>>>>> REPLACE
