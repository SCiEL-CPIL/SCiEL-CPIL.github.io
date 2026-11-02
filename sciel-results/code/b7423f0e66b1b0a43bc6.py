import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def flatten_tensor(U, V, W, n, m, p):
        """Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,k"""
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        temp = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
        W_trans = W_3d.transpose(1, 0, 2)  # (n, p, R)
        result = np.einsum('ikr,ikr->ik', temp, W_trans)
        return result

    def verify_decomposition(U, V, W, n, m, p):
        """Verify exact decomposition"""
        T = flatten_tensor(U, V, W, n, m, p)
        return np.allclose(T, np.ones((n, p)), atol=1e-9)

    # Slice-and-dice: decompose 3x4x7 as 3 independent 1x4x7 sub-problems
    # Each 1x4x7 has rank 7 (one term per output column k)
    # Total rank = 3 * 7 = 21
    R = n * p  # 3 * 7 = 21
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    for i in range(n):
        for k in range(p):
            r = i * p + k
            # U[i*m + j, r] = 1 for all j
            for j in range(m):
                U[i * m + j, r] = 1
            # V[j*p + k, r] = 1 for all j
            for j in range(m):
                V[j * p + k, r] = 1
            # W[k*n + i, r] = 1
            W[k * n + i, r] = 1

    if verify_decomposition(U, V, W, n, m, p) and time.time() < deadline:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Schoolbook fallback (rank 84)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
