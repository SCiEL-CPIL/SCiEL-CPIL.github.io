    while True:
        # Find all edges that are not selected and have no blockers
        mask = (~selected) & (blockers == 0)
        if not mask.any():
            break
        candidates = np.where(mask)[0]
        # To ensure correctness, we process them. 
        # Adding one might block another. 
        # So we iterate.
        for e in candidates:
            if not selected[e] and blockers[e] == 0:
                add_edge(e)
    