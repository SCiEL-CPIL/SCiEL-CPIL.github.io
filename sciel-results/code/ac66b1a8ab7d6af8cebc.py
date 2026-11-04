"""Matrix multiplication 6x6x6 - rank-189 decomposition via block Strassen."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    Ur = U.reshape(n, m, r) if isinstance(U, np.ndarray) else np.array(U).reshape(n, m, r)
    Vr = V.reshape(m, p, r) if isinstance(V, np.ndarray) else np.array(V).reshape(m, p, r)
    Wr = W.reshape(p, n, r) if isinstance(W, np.ndarray) else np.array(W).reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _strassen_2x2():
    """
    Verified rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0] and C[1,1]
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1] and C[1,1]
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0] and C[1,0]
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0] and C[0,1]
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _block_strassen_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row, col] = sU[li*2+lj, c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row, col] = sV[lj*2+lk, c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row, col] = sW[lk*2+li, c]
                    
                    col += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Uses block Strassen: 3x3 blocks of 2x2, each 2x2 using Strassen rank 7.
    Total rank = 27 * 7 = 189.
    """
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Primary: block Strassen (rank 189)
    U_b, V_b, W_b = _block_strassen_6x6x6()
    
    # Verify quickly
    if _verify(U_b, V_b, W_b, 6, 6, 6):
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    # If verification fails, fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
