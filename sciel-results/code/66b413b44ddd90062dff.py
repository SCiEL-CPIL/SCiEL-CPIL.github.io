    S = np.cumsum(seq_pt, axis=0)
    C = np.zeros((n_jobs, n_machines))
    C[:, 0] = S[:, 0]
    
    for m in range(1, n_machines):
        D = C[:, m-1].copy()
        if n_jobs > 1:
            D[1:] -= S[:-1, m]
        # D[0] corresponds to k=0. S_{-1,m} = 0. So D[0] = C[0, m-1]. Correct.
        M = np.maximum.accumulate(D)
        C[:, m] = M + S[:, m]
    return C[-1, -1]
