    newly_covered = balls[c][~covered[balls[c]]]
    if len(newly_covered) > 0:
        covered[newly_covered] = True
        uncovered_count -= len(newly_covered)
        # Get all codewords covering these words
        # cover_map_arr[newly_covered] gives shape (k, 15)
        c2_candidates = cover_map_arr[newly_covered]
        # Flatten and unique
        unique_c2s = np.unique(c2_candidates.flatten())
        cnt[unique_c2s] -= 1
    