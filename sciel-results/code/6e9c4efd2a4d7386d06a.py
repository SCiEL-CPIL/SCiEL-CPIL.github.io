import numpy as np
import time

def construct():
    start = time.time()
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    Uk = np.kron(U2, U2)
    Vk = np.kron(V2, V2)
    Wk = np.kron(W2, W2)
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ih, il = i // 2, i % 2
                jh, jl = j // 2, j % 2
                kh, kl = k // 2, k % 2
                perm_U[i*4+j] = (ih*2+jh)*4 + (il*2+jl)
                perm_V[j*4+k] = (jh*2+kh)*4 + (jl*2+kl)
                perm_W[k*4+i] = (kh*2+ih)*4 + (kl*2+il)
    
    U4 = Uk[perm_U, :]
    V4 = Vk[perm_V, :]
    W4 = Wk[perm_W, :]
    
    # Quick verification
    ok = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = np.dot(np.dot(U4[i*4+j, :], V4[j*4+k, :]), W4[k*4+i, :])
                if s != 1:
                    ok = False
                    break
        if not ok:
            break
    if not ok:
        # Fallback: naive rank 64
        U = [[0]*64 for _ in range(16)]
        V = [[0]*64 for _ in range(16)]
        W = [[0]*64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start > 110:
        U = [[0]*64 for _ in range(16)]
        V = [[0]*64 for _ in range(16)]
        W = [[0]*64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
