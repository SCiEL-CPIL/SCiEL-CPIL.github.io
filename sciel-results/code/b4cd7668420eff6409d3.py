    for r in range(n_restarts):
        # ... greedy ...
        if s is not None:
            s_pruned = fast_prune(s)
            if len(s_pruned) < best_len:
                best_len = len(s_pruned)
                best_sol = s_pruned
    