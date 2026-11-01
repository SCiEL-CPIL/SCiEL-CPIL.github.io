def tensor_check(Uf, Vf, Wf):
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
