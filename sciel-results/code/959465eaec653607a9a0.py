W[(kb*2+0)*n + 0, col] += strassen_W[0][t]  # C[0, kb*2]
W[(kb*2+1)*n + 0, col] += strassen_W[1][t]  # C[0, kb*2+1]
W[(kb*2+0)*n + 1, col] += strassen_W[2][t]  # C[1, kb*2]
W[(kb*2+1)*n + 1, col] += strassen_W[3][t]  # C[1, kb*2+1]
