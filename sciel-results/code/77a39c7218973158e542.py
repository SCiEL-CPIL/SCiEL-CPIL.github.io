<<<<<<< SEARCH
    if best is None:
        # Try n=18
        remaining = deadline - 10 - elapsed()
        if remaining > 5:
            budget = min(20, remaining)
            if budget > 5:
                adj = try_find(18, budget)
                if adj is not None:
                    if _find_bad_subset(adj, 18) is None:
                        edges = _to_edges(adj, 18)
                        best = (18, edges)
>>>>>>> REPLACE

