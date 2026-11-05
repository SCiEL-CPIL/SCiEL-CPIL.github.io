"""Verified low-rank exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    # A: n x m (2x3), B: m x p (3x3), C: n x p (2x3)
    # Partition: A = [A1 | A2] (2x2 | 2x1), B = [B1 | B2] (2x3 | 1x3)
    # C = A1*B1 + A2*B2
    # A1*B1 (2x2 * 2x3): use Strassen on first 2 output cols (7 terms) + direct on 3rd col (4 terms)
    # A2*B2 (2x1 * 1x3): 3 direct terms
    # rank = 7 + 4 + 3 = 14
    rank = 13
    U = [[0] * rank for _ in range(n * m)]   # 6 x 14
    V = [[0] * rank for _ in range(m * p)]   # 9 x 14
    W = [[0] * rank for _ in range(p * n)]   # 6 x 14, W[k*n + i, r]
    col = 0
    def addU(i, j, c, val): U[i * m + j][c] += val
    def addV(j, k, c, val): V[j * p + k][c] += val
    def addW(k, i, c, val): W[k * n + i][c] += val

    # Strassen for P = A1 * B1[:, :2]  (2x2 * 2x2 -> 2x2)
    # A1: A00=U(0,0), A01=U(0,1), A10=U(1,0), A11=U(1,1)
    # B1[:, :2]: B00=V(0,0), B01=V(0,1), B10=V(1,0), B11=V(1,1)
    # P: P00=W(0,0), P01=W(1,0), P10=W(0,1), P11=W(1,1)
    # Standard Strassen Formulas:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    addU(0,0,col,1); addU(1,1,col,1)
    addV(0,0,col,1); addV(1,1,col,1)
    addW(0,0,col,1); addW(1,1,col,1)
    col += 1
    
    # M2 = (A10+A11)B00
    addU(1,0,col,1); addU(1,1,col,1)
    addV(0,0,col,1)
    addW(0,1,col,1)
    col += 1
    
    # M3 = A00(B01-B11)
    addU(0,0,col,1)
    addV(0,1,col,1); addV(1,1,col,-1)
    addW(1,0,col,1); addW(1,1,col,1)
    col += 1
    
    # M4 = A11(B10-B00)
    addU(1,1,col,1)
    addV(1,0,col,1); addV(0,0,col,-1)
    addW(0,0,col,1); addW(0,1,col,1)
    col += 1
    
    # M5 = (A00+A01)B11
    addU(0,0,col,1); addU(0,1,col,1)
    addV(1,1,col,1)
    addW(0,0,col,-1); addW(1,0,col,1)
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    addU(1,0,col,1); addU(0,0,col,-1)
    addV(0,0,col,1); addV(0,1,col,1)
    addW(1,1,col,1)
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    addU(0,1,col,1); addU(1,1,col,-1)
    addV(1,0,col,1); addV(1,1,col,1)
    addW(0,0,col,-1); addW(1,1,col,-1)
    col += 1
    # Direct: C[:,2] from A1 * B[:,2]: C[i,2] = sum_j A[i,j] B[j,2], i in {0,1}, j in {0,1}
    for i in range(2):
        for j in range(2):
            addU(i, j, col, 1)
            addV(j, 2, col, 1)
            addW(2, i, col, 1)
            col += 1
    # A2*B2: C[i,2] += A[i,2]*B[2,2], i in {0,1}  -> 2 terms (not 3, since B2 is 1x3 but A2 is 2x1,
    # C[i,2] = A[i,2]*B[2,2] for each i, that's 2 terms)
    for i in range(2):
        addU(i, 2, col, 1)
        addV(2, 2, col, 1)
        addW(2, i, col, 1)
        col += 1
    # col should equal rank
    assert col == rank, (col, rank)

    # Verify correctness with numpy
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reconstruct tensor: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Expected: C[i,k] = sum_j A[i,j] B[j,k]; tensor T[i,j,k] should be 1 if it's the
    # multiplication tensor, i.e. T[i,j,k] = delta (the multiplication tensor has T[i,j,k]=1 for all i,j,k? 
    # No. The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in the standard basis? 
    # Actually the multiplication tensor has entries T[i,j,k] = 1 for every (i,j,k).
    # Because C_ik = sum_j A_ij B_jk, so the trilinear form is sum_{i,j,k} A_ij B_jk C_ik with coefficient 1.
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Our decomposition gives T_decomp[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    Td = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Td[i, j, k] = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
    if not np.allclose(Td, T, atol=1e-9):
        # Fallback to schoolbook rank n*m*p to guarantee validity
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][c] = 1
                    V2[j*p+k][c] = 1
                    W2[k*n+i][c] = 1
                    c += 1
        return {"U": U2, "V": V2, "W": W2}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
