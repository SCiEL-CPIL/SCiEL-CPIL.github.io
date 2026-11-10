for mask in range(1 << D):
    assignment = np.array([(mask >> i) & 1 for i in range(D)], dtype=bool)
    ml = max_load(assignment)
