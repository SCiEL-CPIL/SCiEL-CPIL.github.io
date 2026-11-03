    for i in range(16):
        u_idx = (i // 4) * 5 + (i % 4)
        for r in range(56):
            U[u_idx][r] += U_44[i][r]
    for j in range(16):
        v_idx = (j // 4) * 5 + (j % 4)
        for r in range(56):
            V[v_idx][r] += V_44[j][r]
    for k in range(16):
        w_idx = (k // 4) * 4 + (k % 4)
        for r in range(56):
            W[w_idx][r] += W_44[k][r]
