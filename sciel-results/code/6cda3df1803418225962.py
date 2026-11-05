<<<<<<< SEARCH
    size = len(code_list)
    counts = np.zeros(total, dtype=np.int32)
    for c in code_list:
        counts[neighbors[c]] += 1
    uncovered_count = int((counts == 0).sum())
=======
    size = len(code_list)
    counts = np.zeros(total, dtype=np.int32)
    for c in code_list:
        counts[neighbors[c]] += 1
    uncovered_count = int((counts == 0).sum())
    initial_code_indices = code_list[:] # Save for fallback
>>>>>>> REPLACE

