import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # We use a Kronecker decomposition: M_{2,4,6} = M_{2,2,2} \otimes M_{1,2,3}
    # Rank = 7 * 6 = 42 (same as current, but we fix the W permutation bug)
    
    # Strassen 2x2x2 (rank 7)
    U1 = np.array([
        [ 1,  0,  1,  0,  1,  1,  0],
        [ 0,  0,  0,  0,  1,  0,  1],
        [ 0,  1,  0,  0,  0, -1,  0],
        [ 1,  1,  0,  1,  0,  0, -1],
    ], dtype=np.int64)

    V1 = np.array([
        [ 1,  1,  0,  0,  0,  1,  0],
        [ 0,  0,  1,  0,  0,  1,  0],
        [ 0,  0,  0,  1,  0,  0,  1],
        [ 1,  0, -1, -1,  1,  0,  1],
    ], dtype=np.int64)

    W1 = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],
        [ 0,  0,  1,  0,  1,  0,  0],
        [ 0,  1,  0,  1,  0,  0,  0],
        [ 1, -1,  1,  0,  0,  1,  0],
    ], dtype=np.int64)

    # Trivial 1x2x3 (rank 6)
    U2 = np.zeros((2, 6), dtype=np.int64)
    U2[0, 0] = 1; U2[1, 3] = 1
    U2[0, 1] = 1; U2[1, 4] = 1
    U2[0, 2] = 1; U2[1, 5] = 1
    
    V2 = np.zeros((6, 6), dtype=np.int64)
    V2[0, 0] = 1; V2[3, 3] = 1
    V2[1, 1] = 1; V2[4, 4] = 1
    V2[2, 2] = 1; V2[5, 5] = 1
    
    W2 = np.zeros((3, 6), dtype=np.int64)
    W2[0, 0] = 1; W2[0, 3] = 1
    W2[1, 1] = 1; W2[1, 4] = 1
    W2[2, 2] = 1; W2[2, 5] = 1

    # Kronecker products
    U_kron = np.kron(U1, U2) # 8x42
    V_kron = np.kron(V1, V2) # 24x42
    W_kron = np.kron(W1, W2) # 12x42
    
    # Permute V to match global B index order: j * 6 + k
    # j = j1 * 2 + j2, k = k1 * 3 + k2
    # Kron V index: v1 * 6 + v2 where v1 = j1*2 + k1, v2 = j2*3 + k2
    V_new = np.zeros_like(V_kron)
    for j in range(4):
        for k in range(6):
            global_idx = j * 6 + k
            j1 = j // 2
            j2 = j % 2
            k1 = k // 3
            k2 = k % 3
            v1 = j1 * 2 + k1
            v2 = j2 * 3 + k2
            kron_idx = v1 * 6 + v2
            V_new[global_idx] = V_kron[kron_idx]

    # Permute W to match global C index order: k * n + i = k * 2 + i
    # i = i1 (since i2=0), k = k1 * 3 + k2
    # Kron W index: w1 * 3 + w2 where w1 = i1*2 + k1, w2 = k2
    W_new = np.zeros_like(W_kron)
    for k in range(6):
        for i in range(2):
            global_idx = k * 2 + i
            i1 = i
            k1 = k // 3
            k2 = k % 3
            w1 = i1 * 2 + k1
            w2 = k2
            kron_idx = w1 * 3 + w2
            W_new[global_idx] = W_kron[kron_idx]

    # U permutation check:
    # Global A index: i * 4 + j = i1 * 4 + j1 * 2 + j2
    # Kron U index: u1 * 2 + u2 where u1 = i1*2 + j1, u2 = j2
    # Kron = (i1*2 + j1)*2 + j2 = 4 i1 + 2 j1 + j2. Matches.

    if time.time() - start_time > 110:
        return {"U": U_kron.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    return {"U": U_kron.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
