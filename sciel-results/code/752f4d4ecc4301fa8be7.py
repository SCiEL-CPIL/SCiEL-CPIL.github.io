def _makespan(perm, pt):
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    prev_row = np.zeros(m, dtype=float)
    pt_local = pt
    for j in range(n):
        job = perm[j]
        curr_row = np.zeros(m, dtype=float)
        # Unroll first step?
        curr_row[0] = prev_row[0] + pt_local[job, 0]
        for k in range(1, m):
            # max(prev_row[k], curr_row[k-1])
            p = prev_row[k]
            c = curr_row[k-1]
            curr_row[k] = (p if p > c else c) + pt_local[job, k]
        prev_row = curr_row
    return prev_row[-1]
