for idx in indices:
    cover_count[idx] -= 1
    if cover_count[idx] == 0:
        uncovered_mask[idx] = True
