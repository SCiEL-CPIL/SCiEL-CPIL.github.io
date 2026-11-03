def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Strategy: split m=4 into two m=2 blocks.
    # For each m=2 block, split p=7 into p=4 and p=3.
    #   p=4: rank 14 (two Strassen 2x2x2)
    #   p=3: rank 11 (one Strassen 2x2x2 + one schoolbook 2x2x1)
    # Rank per m-block: 14 + 11 = 25
    # Total rank: 2 * 25 = 50

    rank_p4 = 14
    rank_p3 = 11
    rank_per_m_block = rank_p4 + rank_p3
    total_rank = 2 * rank_per_m_block # 50

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U_224, V_224, W_224 = _build_224()
    U_223, V_223, W_223 = _build_223()

    col_offset = 0

    for j_start in [0, 2]:
        # p=4 block: k=0..3, rank 14
        # U: i=0,1; j=j_start, j_start+1
        for i in range(2):
            U[i * 4 + j_start, col_offset:col_offset+rank_p4] = U_224[i * 2 + 0, :]
            U[i * 4 + j_start + 1, col_offset:col_offset+rank_p4] = U_224[i * 2 + 1, :]

        # V: j=j_start, j_start+1; k=0..3
        for j_sub in range(2):
            j = j_start + j_sub
            for k in range(4):
                V[j * 7 + k, col_offset:col_offset+rank_p4] = V_224[j_sub * 2 + k, :]

        # W: k=0..3; i=0,1
        for k in range(4):
            for i in range(2):
                W[k * 2 + i, col_offset:col_offset+rank_p4] = W_224[k * 2 + i, :]

        col_offset += rank_p4

        # p=3 block: k=4..6, rank 11
        for i in range(2):
            U[i * 4 + j_start, col_offset:col_offset+rank_p3] = U_223[i * 2 + 0, :]
            U[i * 4 + j_start + 1, col_offset:col_offset+rank_p3] = U_223[i * 2 + 1, :]

        for j_sub in range(2):
            j = j_start + j_sub
            for k in range(3):
                V[j * 7 + 4 + k, col_offset:col_offset+rank_p3] = V_223[j_sub * 2 + k, :]

        for k in range(3):
            for i in range(2):
                W[(4 + k) * 2 + i, col_offset:col_offset+rank_p3] = W_223[k * 2 + i, :]

        col_offset += rank_p3

    # Verify and return
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback: schoolbook rank 56
            r = n * m * p
            U_fb = np.zeros((n * m, r))
            V_fb = np.zeros((m * p, r))
            W_fb = np.zeros((p * n, r))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
