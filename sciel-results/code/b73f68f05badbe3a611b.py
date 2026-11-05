diffs = np.diff(sorted_windows, axis=1)
distinct_counts = 1 + (diffs != 0).sum(axis=1)
