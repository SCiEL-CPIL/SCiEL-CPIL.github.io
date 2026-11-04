import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_laderman_4x4():
    """
    Build the 4x4 Laderman decomposition (rank 20).
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    I2 = np.eye(2)
    
    def set_block(mat, mat_type, bi, bj, coeffs, term):
        for ii in range(2):
            for jj in range(2):
                if mat_type == 'U':
                    row = (bi*2+ii)*4 + (bj*2+jj)
                elif mat_type == 'V':
                    row = (bi*2+ii)*4 + (bj*2+jj)
                elif mat_type == 'W':
                    row = (bj*2+jj)*4 + (bi*2+ii)
                mat[row, term] = coeffs[ii, jj]
    
    # Term 1
    set_block(U, 'U', 0, 0, I2, 0); set_block(U, 'U', 1, 1, I2, 0)
    set_block(V, 'V', 0, 0, I2, 0); set_block(V, 'V', 1, 1, I2, 0)
    set_block(W, 'W', 0, 0, I2, 0); set_block(W, 'W', 1, 1, I2, 0)
    
    # Term 2
    set_block(U, 'U', 1, 0, I2, 1); set_block(U, 'U', 1, 1, I2, 1)
    set_block(V, 'V', 0, 0, I2, 1)
    set_block(W, 'W', 1, 0, I2, 1); set_block(W, 'W', 1, 1, I2, 1)
    
    # Term 3
    set_block(U, 'U', 0, 0, I2, 2)
    set_block(V, 'V', 0, 1, I2, 2); set_block(V, 'V', 1, 1, -I2, 2)
    set_block(W, 'W', 0, 1, I2, 2); set_block(W, 'W', 1, 1, I2, 2)
    
    # Term 4
    set_block(U, 'U', 1, 1, I2, 3)
    set_block(V, 'V', 1, 0, I2, 3); set_block(V, 'V', 0, 0, -I2, 3)
    set_block(W, 'W', 0, 0, I2, 3); set_block(W, 'W', 1, 0, I2, 3)
    
    # Term 5
    set_block(U, 'U', 0, 0, I2, 4); set_block(U, 'U', 0, 1, I2, 4)
    set_block(V, 'V', 1, 1, I2, 4)
    set_block(W, 'W', 0, 0, I2, 4); set_block(W, 'W', 0, 1, -I2, 4)
    
    # Term 6
    set_block(U, 'U', 1, 0, I2, 5); set_block(U, 'U', 0, 0, -I2, 5)
    set_block(V, 'V', 0, 0, I2, 5); set_block(V, 'V', 0, 1, I2, 5)
    set_block(W, 'W', 1, 1, I2, 5)
    
    # Term 7
    set_block(U, 'U', 0, 1, I2, 6); set_block(U, 'U', 1, 1, -I2, 6)
    set_block(V, 'V', 1, 0, I2, 6); set_block(V, 'V', 1, 1, I2, 6)
    set_block(W, 'W', 0, 0, I2, 6)
    
    # Term 8
    set_block(U, 'U', 0, 0, I2, 7); set_block(U, 'U', 1, 0, I2, 7)
    set_block(V, 'V', 1, 1, I2, 7); set_block(V, 'V', 0, 1, I2, 7)
    set_block(W, 'W', 0, 0, I2, 7); set_block(W, 'W', 0, 1, I2, 7)
    
    # Term 9
    set_block(U, 'U', 0, 1, I2, 8); set_block(U, 'U', 1, 1, I2, 8)
    set_block(V, 'V', 1, 0, I2, 8); set_block(V, 'V', 0, 0, I2, 8)
    set_block(W, 'W', 0, 1, I2, 8); set_block(W, 'W', 1, 1, I2, 8)
    
    # Term 10
    set_block(U, 'U', 1, 0, I2, 9); set_block(U, 'U', 0, 0, I2, 9)
    set_block(V, 'V', 0, 0, I2, 9); set_block(V, 'V', 1, 0, I2, 9)
    set_block(W, 'W', 1, 0, I2, 9); set_block(W, 'W', 0, 0, I2, 9)
    
    # Term 11
    set_block(U, 'U', 1, 1, I2, 10); set_block(U, 'U', 0, 1, I2, 10)
    set_block(V, 'V', 0, 1, I2, 10); set_block(V, 'V', 1, 1, I2, 10)
    set_block(W, 'W', 1, 1, I2, 10); set_block(W, 'W', 0, 1, I2, 10)
    
    # Term 12
    set_block(U, 'U', 0, 0, I2, 11); set_block(U, 'U', 1, 1, -I2, 11)
    set_block(U, 'U', 0, 1, I2, 11); set_block(U, 'U', 1, 0, -I2, 11)
    set_block(V, 'V', 0, 0, I2, 11); set_block(V, 'V', 1, 1, -I2, 11)
    set_block(V, 'V', 0, 1, I2, 11); set_block(V, 'V', 1, 0, -I2, 11)
    set_block(W, 'W', 0, 0, I2, 11); set_block(W, 'W', 1, 1, I2, 11)
    set_block(W, 'W', 0, 1, -I2, 11); set_block(W, 'W', 1, 0, -I2, 11)
    
    # Term 13
    set_block(U, 'U', 0, 0, I2, 12); set_block(U, 'U', 1, 1, I2, 12)
    set_block(U, 'U', 0, 1, -I2, 12); set_block(U, 'U', 1, 0, -I2, 12)
    set_block(V, 'V', 0, 0, I2, 12); set_block(V, 'V', 1, 1, I2, 12)
    set_block(V, 'V', 0, 1, -I2, 12); set_block(V, 'V', 1, 0, -I2, 12)
    set_block(W, 'W', 0, 0, I2, 12); set_block(W, 'W', 1, 1, I2, 12)
    set_block(W, 'W', 0, 1, I2, 12); set_block(W, 'W', 1, 0, I2, 12)
    
    # Term 14
    set_block(U, 'U', 0, 0, I2, 13); set_block(U, 'U', 0, 1, -I2, 13)
    set_block(U, 'U', 1, 0, I2, 13); set_block(U, 'U', 1, 1, -I2, 13)
    set_block(V, 'V', 0, 0, I2, 13); set_block(V, 'V', 0, 1, -I2, 13)
    set_block(V, 'V', 1, 0, I2, 13); set_block(V, 'V', 1, 1, -I2, 13)
    set_block(W, 'W', 0, 0, I2, 13); set_block(W, 'W', 0, 1, -I2, 13)
    set_block(W, 'W', 1, 0, I2, 13); set_block(W, 'W', 1, 1, -I2, 13)
    
    # Term 15
    set_block(U, 'U', 0, 0, I2, 14); set_block(U, 'U', 0, 1, I2, 14)
    set_block(U, 'U', 1, 0, -I2, 14); set_block(U, 'U', 1, 1, -I2, 14)
    set_block(V, 'V', 0, 0, I2, 14); set_block(V, 'V', 0, 1, I2, 14)
    set_block(V, 'V', 1, 0, -I2, 14); set_block(V, 'V', 1, 1, -I2, 14)
    set_block(W, 'W', 0, 0, I2, 14); set_block(W, 'W', 0, 1, I2, 14)
    set_block(W, 'W', 1, 0, -I2, 14); set_block(W, 'W', 1, 1, -I2, 14)
    
    # Term 16
    set_block(U, 'U', 1, 0, I2, 15); set_block(U, 'U', 1, 1, -I2, 15)
    set_block(V, 'V', 0, 0, I2, 15); set_block(V, 'V', 1, 1, -I2, 15)
    set_block(W, 'W', 1, 0, I2, 15); set_block(W, 'W', 1, 1, -I2, 15)
    
    # Term 17
    set_block(U, 'U', 0, 1, I2, 16); set_block(U, 'U', 0, 0, -I2, 16)
    set_block(V, 'V', 1, 0, I2, 16); set_block(V, 'V', 1, 1, -I2, 16)
    set_block(W, 'W', 0, 1, I2, 16); set_block(W, 'W', 0, 0, -I2, 16)
    
    # Term 18
    set_block(U, 'U', 1, 0, I2, 17); set_block(U, 'U', 0, 0, -I2, 17)
    set_block(V, 'V', 0, 1, I2, 17); set_block(V, 'V', 1, 1, -I2, 17)
    set_block(W, 'W', 1, 0, I2, 17); set_block(W, 'W', 1, 1, -I2, 17)
    
    # Term 19
    set_block(U, 'U', 0, 1, I2, 18); set_block(U, 'U', 1, 1, -I2, 18)
    set_block(V, 'V', 1, 0, I2, 18); set_block(V, 'V', 0, 0, -I2, 18)
    set_block(W, 'W', 0, 1, I2, 18); set_block(W, 'W', 0, 0, -I2, 18)
    
    # Term 20
    set_block(U, 'U', 0, 0, I2, 19); set_block(U, 'U', 1, 1, I2, 19)
    set_block(U, 'U', 0, 1, I2, 19); set_block(U, 'U', 1, 0, I2, 19)
    set_block(V, 'V', 0, 0, I2, 19); set_block(V, 'V', 1, 1, I2, 19)
    set_block(V, 'V', 0, 1, I2, 19); set_block(V, 'V', 1, 0, I2, 19)
    set_block(W, 'W', 0, 0, I2, 19); set_block(W, 'W', 1, 1, I2, 19)
    set_block(W, 'W', 0, 1, I2, 19); set_block(W, 'W', 1, 0, I2, 19)
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS

    U22, V22, W22 = _build_strassen_2x2()
    U44, V44, W44 = _build_laderman_4x4()

    # Verify 4x4 Laderman
    np.random.seed(123)
    A4 = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    B4 = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    C4_expected = A4 @ B4
    C4_computed = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            s = 0.0
            for j in range(4):
                for r in range(20):
                    s += A4[i, j] * B4[j, k] * U44[i*4+j, r] * V44[j*4+k, r] * W44[k*4+i, r]
            C4_computed[i, k] = s

    if not np.allclose(C4_computed, C4_expected, atol=1e-9):
        return _construct_fallback(n, m, p, deadline)

    rank = 20 * 7
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for r44 in range(20):
        for r22 in range(7):
            r_final = r44 * 7 + r22

            for bi in range(4):
                for bj in range(4):
                    i_start = bi * 2
                    j_start = bj * 2
                    if i_start >= 6 or j_start >= 6:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            i = bi * 2 + ii
                            j = bj * 2 + jj
                            if i >= 6 or j >= 6:
                                continue
                            u44_val = U44[i * 4 + j, r44]
                            if u44_val == 0:
                                continue
                            u22_val = U22[ii * 2 + jj, r22]
                            if u22_val == 0:
                                continue
                            U66[i * 6 + j, r_final] = u44_val * u22_val

            for bj in range(4):
                for bk in range(4):
                    j_start = bj * 2
                    k_start = bk * 2
                    if j_start >= 6 or k_start >= 6:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            j = bj * 2 + jj
                            k = bk * 2 + kk
                            if j >= 6 or k >= 6:
                                continue
                            v44_val = V44[j * 4 + k, r44]
                            if v44_val == 0:
                                continue
                            v22_val = V22[jj * 2 + kk, r22]
                            if v22_val == 0:
                                continue
                            V66[j * 6 + k, r_final] = v44_val * v22_val

            for bi in range(4):
                for bk in range(4):
                    i_start = bi * 2
                    k_start = bk * 2
                    if i_start >= 6 or k_start >= 6:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            i = bi * 2 + ii
                            k = bk * 2 + kk
                            if i >= 6 or k >= 6:
                                continue
                            w44_val = W44[k * 4 + i, r44]
                            if w44_val == 0:
                                continue
                            w22_val = W22[kk * 2 + ii, r22]
                            if w22_val == 0:
                                continue
                            W66[k * 6 + i, r_final] = w44_val * w22_val

    # Prune zero columns
    mask = np.any(U66 != 0, axis=0) & np.any(V66 != 0, axis=0) & np.any(W66 != 0, axis=0)
    U66 = U66[:, mask]
    V66 = V66[:, mask]
    W66 = W66[:, mask]
    
    # Verify
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(U66.shape[1]):
                        s += A_test[i, j] * B_test[j, k] * U66[i*m + j, r] * V66[j*p + k, r] * W66[k*n + i, r]
                C_computed[i, k] = s

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return _construct_fallback(n, m, p, deadline)

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

def _construct_fallback(n, m, p, deadline):
    # Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
