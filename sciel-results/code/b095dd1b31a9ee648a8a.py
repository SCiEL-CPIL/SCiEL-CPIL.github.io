    for k in range(16):
        j1 = k // 8
        j2 = (k // 4) % 2
        i1 = (k // 2) % 2
        i2 = k % 2
        L = 8 * i1 + 4 * j1 + 2 * i2 + j2
        W[k, :] = Kron_W[L, :]
