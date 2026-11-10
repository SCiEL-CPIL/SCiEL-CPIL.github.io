def init_state(S):
    global cnt, F, cover
    cnt = np.zeros((2401, 50), dtype=np.int32)
    for d in range(50):
        cnt[:, d] = S[lines[:, d, :]].sum(axis=1)
    F = np.zeros(2401, dtype=np.int64)
    for d in range(50):
        F += (cnt[:, d] == 0).astype(np.int64) * (1 << d)
    cover = np.zeros(2401, dtype=np.int32)
    for d in range(50):
        crit_x = np.where(F == (1 << d))[0]
        for x in crit_x:
            cover[lines[x, d]] += 1
