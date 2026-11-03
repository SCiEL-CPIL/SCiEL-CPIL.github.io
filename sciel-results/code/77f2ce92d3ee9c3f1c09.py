"""Strassen-based rank-56 decomposition for 4x4 * 4x8 -> 4x8.

Uses Strassen's 7-multiplication algorithm on 2x2 blocks recursively.
4x4x4: 4 independent 2x2x2 Strassen blocks -> rank 28
4x4x8: two 4x4x4 blocks -> rank 56
"""

# EVOLVE-BLOCK-START

import time
import numpy as np


def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 -> 2x2 Strassen decomposition.
    
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = delta_{i,k} ... 
    Actually for 2x2*2x2->2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor: T[i,j,k] = 1 for all i,j,k in {0,1}
    Decomposition: sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all i,j,k
    
    Strassen 7 bilinear forms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4
    C10 = M2 + M7
    C11 = M1 - M3 + M6 - M7
    """
    # U: rows indexed by i*2+j for A[i,j]
    # V: rows indexed by j*2+k for B[j,k]  
    # W: rows indexed by k*2+i for C[k,i] -> but C is 2x2, C[k,i] means row k, col i
    # Wait: W[k*n+i] where n=2, so W[k*2+i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00 = U[0], A11 = U[3], B00 = V[0], B11 = V[3]
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    # C00 += M1, C11 += M1
    # C00: k=0,i=0 -> W[0*2+0]=W[0]
    # C11: k=1,i=1 -> W[1*2+1]=W[3]
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (A10+A11)B00
    # A10 = U[2], A11 = U[3], B00 = V[0]
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    # C10 += M2: k=1,i=0 -> W[1*2+0]=W[2]
    W[2, 1] += 1
    
    # M3 = A00(B10-B11)
    # A00 = U[0], B10 = V[2], B11 = V[3]
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    # C01 += M3: k=0,i=1 -> W[0*2+1]=W[1]
    # C11 -= M3: k=1,i=1 -> W[3]
    W[1, 2] += 1; W[3, 2] -= 1
    
    # M4 = A11(B01-B10)
    # A11 = U[3], B01 = V[1], B10 = V[2]
    U[3, 3] += 1
    V[1, 3] += 1; V[2, 3] -= 1
    # C00 -= M4: W[0]
    # C01 += M4: W[1]
    W[0, 3] -= 1; W[1, 3] += 1
    
    # M5 = (A00+A01)B11
    # A00 = U[0], A01 = U[1], B11 = V[3]
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    # C00 += M5: W[0]
    W[0, 4] += 1
    
    # M6 = (A10-A00)(B00+B10)
    # A10 = U[2], A00 = U[0], B00 = V[0], B10 = V[2]
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[2, 5] += 1
    # C11 += M6: W[3]
    W[3, 5] += 1
    
    # M7 = (A01-A11)(B01+B11)
    # A01 = U[1], A11 = U[3], B01 = V[1], B11 = V[3]
    U[1, 6] += 1; U[3, 6] -= 1
    V[1, 6] += 1; V[3, 6] += 1
    # C00 -= M7: W[0]
    # C10 += M7: W[2]
    # C11 -= M7: W[3]
    W[0, 6] -= 1; W[2, 6] += 1; W[3, 6] -= 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 * 4x4 -> 4x4 using 4 Strassen 2x2 blocks."""
    n, m, p = 4, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    # 4 blocks: (i_block, j_block) where i_block in {0,1}, j_block in {0,1}
    # Block (bi, bj) handles A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, 2*bj:2*bj+2] -> C[2*bi:2*bi+2, 2*bj:2*bj+2]
    for bi in range(2):
        for bj in range(2):
            base_col = (bi * 2 + bj) * 7
            for c in range(7):
                col = base_col + c
                # U: A rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                for di in range(2):
                    for dj in range(2):
                        a_row = (2 * bi + di) * m + (2 * bj + dj)
                        u_val = U2[di * 2 + dj, c]
                        if u_val != 0:
                            U[a_row, col] += u_val
                # V: B rows 2*bj..2*bj+1, cols 2*bj..2*bj+1
                for dj in range(2):
                    for dk in range(2):
                        b_row = (2 * bj + dj) * p + (2 * bj + dk)
                        v_val = V2[dj * 2 + dk, c]
                        if v_val != 0:
                            V[b_row, col] += v_val
                # W: C rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # W[k*n+i] where k is row of C, i is col of C
                for dk in range(2):
                    for di in range(2):
                        c_row = (2 * bi + dk) * n + (2 * bj + di)
                        w_val = W2[dk * 2 + di, c]
                        if w_val != 0:
                            W[c_row, col] += w_val
    
    return U, V, W


def _build_4x4x8_strassen():
    """Build rank-56 decomposition for 4x4 * 4x8 -> 4x8.
    
    Split B into two 4x4 blocks (cols 0-3 and 4-7), C into two 4x4 blocks.
    Each half uses a 4x4x4 Strassen decomposition of rank 28.
    Total rank = 56.
    """
    n, m, p = 4, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # First half: B cols 0-3, C rows 0-3
    for i in range(16):
        for c in range(28):
            U[i, c] = U4[i, c]
            V[i, c] = V4[i, c]
            W[i, c] = W4[i, c]
    
    # Second half: B cols 4-7, C rows 4-7
    for i in range(16):
        for c in range(28):
            U[i, c + 28] = U4[i, c]
            # V: B rows j, cols 4-7 -> V[j*8 + (4+k)]
            # V4 has rows j*4+k for k in 0..3, so V[j*8+4+k] = V4[j*4+k]
            V[i + 16, c + 28] = V4[i, c]  # Wait, V4 row index is j*4+k
            # Actually V4 has shape (16, 28), row = j*4+k
            # For second half, B row j, col 4+k -> global row j*8 + 4 + k
            # So V[j*8 + 4 + k, c+28] = V4[j*4+k, c]
            # Let me redo this properly
            pass
    
    # Let me rebuild more carefully
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # First half: B cols 0-3, C rows 0-3
    for i in range(16):
        for c in range(28):
            U[i, c] = U4[i, c]
            V[i, c] = V4[i, c]
            W[i, c] = W4[i, c]
    
    # Second half: B cols 4-7, C rows 4-7
    for i in range(16):
        for c in range(28):
            U[i, c + 28] = U4[i, c]
            # V4 row i corresponds to B[j, k] where i = j*4+k
            # For second half, B[j, 4+k] -> global V row = j*8 + 4 + k
            j, k = divmod(i, 4)
            v_row = j * p + 4 + k
            V[v_row, c + 28] = V4[i, c]
            # W4 row i corresponds to C[k, i_c] where i = k*4+i_c (k=row, i_c=col)
            # For second half, C[4+k, i_c] -> global W row = (4+k)*n + i_c
            k_c, i_c = divmod(i, 4)
            w_row = (4 + k_c) * n + i_c
            W[w_row, c + 28] = W4[i, c]
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False, (i, j, k, s)
    return True, None


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 8
    
    # Try Strassen rank-56
    try:
        U, V, W = _build_4x4x8_strassen()
        ok, err = _verify(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            print(f"Strassen verification failed: {err}")
    except Exception as e:
        print(f"Strassen construction failed: {e}")
    
    # Fallback to schoolbook
    try:
        U, V, W = _schoolbook(n, m, p)
        ok, err = _verify(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        print(f"Schoolbook failed: {e}")
    
    # Last resort: return schoolbook without verification
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
