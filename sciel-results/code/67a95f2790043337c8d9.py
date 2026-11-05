    for r in range(101):
        if r in subsets_by_residue:
            # Pick the one with smallest size
            best = min(subsets_by_residue[r], key=lambda x: x[0])
            best_assignment[r] = best[1]
