    # Sort candidates by span descending to prioritize long shortcuts
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    # Recalculate conflicts after sorting?
    # The conflicts are based on indices. If we sort, indices change.
    # So we must generate conflicts AFTER sorting.
