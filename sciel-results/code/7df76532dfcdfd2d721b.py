"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a recursive Strassen-based decomposition with rank 98.
    """
    start = time.time()
    deadline = start + 100  # Safety margin for 120s limit
    n, m, p = DIMENSIONS

    # Schoolbook rank-128 as fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2 factors
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    # Build rank-98 decomposition for 4x4 * 4x8
    # Split B into two 4x4 blocks
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(2):
        for s in range(7):
            u_block, v_block, w_block = S[s]
            for r in range(7):
                u_scalar, v_scalar, w_scalar = S[r]
                col = t * 49 + s * 7 + r

                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei*2+ej]
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                j = 2*bj + ej
                                U[i*m + j][col] += cb * cs

                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei*2+ej]
                                if cs == 0:
                                    continue
                                j = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                V[j*p + k][col] += cb * cs

                for bi in range(2):
                    for bj in range(2):
                        cb = w_block[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei*2+ej]
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                W[k*n + i][col] += cb * cs

    # Verify
    def _verify(U_mat, V_mat, W_mat):
        rk = len(U_mat[0])
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        U3 = U_arr.reshape(n, m, rk)
        V3 = V_arr.reshape(m, p, rk)
        W3 = W_arr.reshape(p, n, rk)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)

    if time.time() < deadline:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    return _schoolbook()
