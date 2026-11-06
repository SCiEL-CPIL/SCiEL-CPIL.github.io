    colour_counts = np.zeros(14, dtype=np.int32)
    for c in colours:
        colour_counts[c] += 1
    num_distinct = np.sum(colour_counts > 0)
