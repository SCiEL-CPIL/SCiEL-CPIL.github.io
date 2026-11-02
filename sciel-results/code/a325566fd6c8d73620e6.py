"""Tensor decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-10):
    """Check that sum_r U[:,r] V[:,r] W[:,r] equals the matrix multiplication tensor."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Ua: (n*m, rank), Va: (m*p, rank), Wa: (p*n, rank)
    # Tensor T[a_idx, b_idx, c_idx] = sum_r Ua[a_idx,r] * Va[b_idx,r] * Wa[c_idx,r]
    # a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    rank = Ua.shape[1]
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    if np.max(np.abs(T - T_target)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Rank 26 decomposition:
    # 7 Strassen terms for the 2x2 core (indices 0,1)
    # 15 border terms for the remaining interactions
    # 4 additional terms for A02*B20, A02*B21, A12*B20, A12*B21
    
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 26
    V = [[0.0] * rank for _ in range(m * p)]  # 9 x 26
    W = [[0.0] * rank for _ in range(p * n)]  # 9 x 26
    
    # Index conventions:
    # U row: i*m + j (i = row of A, j = col of A)
    # V row: j*p + k (j = row of B, k = col of B)
    # W row: k*n + i (k = col of C, i = row of C)
    
    # --- Strassen 2x2 core (rows/cols 0,1 of A,B -> C[0,0],C[0,1],C[1,0],C[1,1]) ---
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # A00=U[0], A11=U[4], B00=V[0], B11=V[4]
    # C00=W[0], C11=W[4]
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2
    # A10=U[3], A11=U[4], B00=V[0]
    # C10=W[3]
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1
    
    # M3 = A00(B10-B11) -> C00 -= M3
    # A00=U[0], B10=V[3], B11=V[4]
    # C00=W[0]
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[0][2] = -1
    
    # M4 = A11(B01-B00) -> C10 += M4
    # A11=U[4], B01=V[1], B00=V[0]
    # C10=W[3]
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
    # A00=U[0], A01=U[1], B11=V[4]
    # C01=W[1], C11=W[4]
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C00 += M6
    # A10=U[3], A00=U[0], B00=V[0], B01=V[1]
    # C00=W[0]
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 -= M7, C01 += M7
    # A01=U[1], A11=U[4], B10=V[3], B11=V[4]
    # C00=W[0], C01=W[1]
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    # --- Border terms: row 2 of C (i=2) ---
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # C20: W[0*3+2]=W[2]
    U[6][7] = 1   # A20
    V[0][7] = 1   # B00
    W[2][7] = 1
    
    U[7][8] = 1   # A21
    V[3][8] = 1   # B10
    W[2][8] = 1
    
    U[8][9] = 1   # A22
    V[6][9] = 1   # B20
    W[2][9] = 1
    
    # C21: W[1*3+2]=W[5]
    U[6][10] = 1  # A20
    V[1][10] = 1  # B01
    W[5][10] = 1
    
    U[7][11] = 1  # A21
    V[4][11] = 1  # B11
    W[5][11] = 1
    
    U[8][12] = 1  # A22
    V[7][12] = 1  # B21
    W[5][12] = 1
    
    # C22: W[2*3+2]=W[8]
    U[6][13] = 1  # A20
    V[2][13] = 1  # B02
    W[8][13] = 1
    
    U[7][14] = 1  # A21
    V[5][14] = 1  # B12
    W[8][14] = 1
    
    U[8][15] = 1  # A22
    V[8][15] = 1  # B22
    W[8][15] = 1
    
    # --- Border terms: col 2 of C (k=2) ---
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # C02: W[2*3+0]=W[6]
    U[0][16] = 1  # A00
    V[2][16] = 1  # B02
    W[6][16] = 1
    
    U[1][17] = 1  # A01
    V[5][17] = 1  # B12
    W[6][17] = 1
    
    U[2][18] = 1  # A02
    V[8][18] = 1  # B22
    W[6][18] = 1
    
    # C12: W[2*3+1]=W[7]
    U[3][19] = 1  # A10
    V[2][19] = 1  # B02
    W[7][19] = 1
    
    U[4][20] = 1  # A11
    V[5][20] = 1  # B12
    W[7][20] = 1
    
    U[5][21] = 1  # A12
    V[8][21] = 1  # B22
    W[7][21] = 1
    
    # --- Missing cross terms: A col 2 with B col 2 contributing to C[0:2, 0:2] ---
    # C00 += A02*B20
    # C01 += A02*B21
    # C10 += A12*B20
    # C11 += A12*B21
    
    # A02*B20 -> C00: W[0]
    U[2][22] = 1   # A02
    V[6][22] = 1   # B20
    W[0][22] = 1
    
    # A02*B21 -> C01: W[1]
    U[2][23] = 1   # A02
    V[7][23] = 1   # B21
    W[1][23] = 1
    
    # A12*B20 -> C10: W[3]
    U[5][24] = 1   # A12
    V[6][24] = 1   # B20
    W[3][24] = 1
    
    # A12*B21 -> C11: W[4]
    U[5][25] = 1   # A12
    V[7][25] = 1   # B21
    W[4][25] = 1
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: brute-force rank 27
    rank27 = 27
    U27 = [[0.0] * rank27 for _ in range(n * m)]
    V27 = [[0.0] * rank27 for _ in range(m * p)]
    W27 = [[0.0] * rank27 for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i * m + j][r] = 1.0
                V27[j * p + k][r] = 1.0
                W27[k * n + i][r] = 1.0
                r += 1
    
    if _verify(U27, V27, W27):
        return {"U": U27, "V": V27, "W": W27}
    
    # Last resort: return the rank-26 anyway (should not happen)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
