for i in range(n):
    for j in range(m):
        for k in range(p):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            s = 0.0
            for t in range(rank):
                s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
            diff = abs(s - 1.0)
