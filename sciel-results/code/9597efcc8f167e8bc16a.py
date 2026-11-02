"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T_ijk = sum_r U_ijk,r V_jk,r W_ki,r
    # We want T_ijk = 1 for all i,j,k
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # r=1: M1 = (A00+A11)(B00+B11)
    # C00 += 1*M1, C11 += 1*M1
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11
    V[0, 0] = 1; V[3, 0] = 1 # B00, B11
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # r=2: M2 = (A10+A11)B00
    # C10 += 1*M2, C11 += -1*M2
    U[2, 1] = 1; U[3, 1] = 1 # A10, A11
    V[0, 1] = 1 # B00
    W[2, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # r=3: M3 = A00(B01-B11)
    # C01 += 1*M3, C11 += 1*M3
    U[0, 2] = 1 # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, B11
    W[1, 2] = 1; W[3, 2] = 1 # C01, C11
    
    # r=4: M4 = A11(B10-B00)
    # C00 += 1*M4, C10 += 1*M4
    U[3, 3] = 1 # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, B00
    W[0, 3] = 1; W[2, 3] = 1 # C00, C10
    
    # r=5: M5 = (A00+A01)B11
    # C00 += -1*M5, C01 += 1*M5
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01
    V[3, 4] = 1 # B11
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # r=6: M6 = (A10-A00)(B00+B01)
    # C11 += 1*M6
    U[2, 5] = 1; U[0, 5] = -1 # A10, A00
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    W[3, 5] = 1 # C11
    
    # r=7: M7 = (A01-A11)(B10+B11)
    # C00 += 1*M7
    U[1, 6] = 1; U[3, 6] = -1 # A01, A11
    V[2, 6] = 1; V[3, 6] = 1 # B10, B11
    W[0, 6] = 1 # C00
    
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_444():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using recursive Strassen.
    Rank 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_222()
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # Strassen 4x4x4 splits A, B into 2x2 blocks of 2x2 matrices.
    # The 7 products M1..M7 are 2x2x2 MMs.
    # For each of the 7 Strassen terms (r_super), we have a 2x2x2 MM.
    # Each 2x2x2 MM has rank 7 (r_inner).
    # Total rank 49.
    
    # We need to map the indices.
    # A is 4x4. Blocks A00, A01, A10, A11.
    # A00 is A[0:2, 0:2]. Indices i in 0,1; j in 0,1.
    # A01 is A[0:2, 2:4]. Indices i in 0,1; j in 2,3.
    # A10 is A[2:4, 0:2]. Indices i in 2,3; j in 0,1.
    # A11 is A[2:4, 2:4]. Indices i in 2,3; j in 2,3.
    
    # Similarly for B and C.
    
    # Strassen formulas for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix.
    # M_r = C_r (2x2).
    # We decompose M_r using U2, V2, W2.
    # M_r[i2, j2] = sum_k U2[i2*m2+j2, k] V2[j2*p2+k2, k] W2[k2*n2+i2, k]
    # Here i2, j2, k2 are local indices 0,1.
    
    # We need to map local indices to global indices.
    # For A:
    # Block 00: i=0,1, j=0,1 -> global i=i2, j=j2
    # Block 01: i=0,1, j=2,3 -> global i=i2, j=2+j2
    # Block 10: i=2,3, j=0,1 -> global i=2+i2, j=j2
    # Block 11: i=2,3, j=2,3 -> global i=2+i2, j=2+j2
    
    # For B:
    # Block 00: j=0,1, k=0,1 -> global j=j2, k=k2
    # Block 01: j=0,1, k=2,3 -> global j=j2, k=2+k2
    # Block 10: j=2,3, k=0,1 -> global j=2+j2, k=k2
    # Block 11: j=2,3, k=2,3 -> global j=2+j2, k=2+k2
    
    # For C:
    # Block 00: k=0,1, i=0,1 -> global k=k2, i=i2
    # Block 01: k=0,1, i=2,3 -> global k=k2, i=2+i2
    # Block 10: k=2,3, i=0,1 -> global k=2+k2, i=i2
    # Block 11: k=2,3, i=2,3 -> global k=2+k2, i=2+i2
    
    # Coefficients for A blocks in each M:
    # M1: A00(+1), A11(+1)
    # M2: A10(+1), A11(+1)
    # M3: A00(+1)
    # M4: A11(+1)
    # M5: A00(+1), A01(+1)
    # M6: A10(+1), A00(-1)
    # M7: A01(+1), A11(-1)
    
    # Coefficients for B blocks:
    # M1: B00(+1), B11(+1)
    # M2: B00(+1)
    # M3: B01(+1), B11(-1)
    # M4: B10(+1), B00(-1)
    # M5: B11(+1)
    # M6: B00(+1), B01(+1)
    # M7: B10(+1), B11(+1)
    
    # Coefficients for C blocks:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    # Helper to add block
    def add_block(U_block, V_block, W_block, coeff_A, coeff_B, coeff_C, base_idx):
        # base_idx is the column offset in U, V, W (0 to 48)
        for r_inner in range(7):
            col = base_idx + r_inner
            for i2 in range(2):
                for j2 in range(2):
                    # U
                    if coeff_A != 0:
                        # Determine global i, j based on block
                        pass # Handled by mapping below
            
            # This loop is getting complex. Let's do it explicitly for each M.
            
    # Explicit mapping
    # M1: A00, A11; B00, B11; C00, C11
    # M1 uses U2, V2, W2.
    # For each r_inner in 0..6:
    #   col = r_inner
    #   U[:, col] gets contributions from A00 and A11.
    #   A00 is at rows 0..3 (i=0,1; j=0,1). U2 rows 0..3.
    #   A11 is at rows 12..15 (i=2,3; j=2,3). U2 rows 0..3 mapped to 12..15.
    
    for r_super in range(7):
        base = r_super * 7
        # Determine which A blocks and coeffs
        A_blocks = []
        B_blocks = []
        C_blocks = []
        
        if r_super == 0: # M1
            A_blocks = [(0,0,1), (1,1,1)]
            B_blocks = [(0,0,1), (1,1,1)]
            C_blocks = [(0,0,1), (1,1,1)]
        elif r_super == 1: # M2
            A_blocks = [(1,0,1), (1,1,1)]
            B_blocks = [(0,0,1)]
            C_blocks = [(1,0,1), (1,1,-1)]
        elif r_super == 2: # M3
            A_blocks = [(0,0,1)]
            B_blocks = [(0,1,1), (1,1,-1)]
            C_blocks = [(0,1,1), (1,1,1)]
        elif r_super == 3: # M4
            A_blocks = [(1,1,1)]
            B_blocks = [(1,0,1), (0,0,-1)]
            C_blocks = [(0,0,1), (1,0,1)]
        elif r_super == 4: # M5
            A_blocks = [(0,0,1), (0,1,1)]
            B_blocks = [(1,1,1)]
            C_blocks = [(0,0,-1), (0,1,1)]
        elif r_super == 5: # M6
            A_blocks = [(1,0,1), (0,0,-1)]
            B_blocks = [(0,0,1), (0,1,1)]
            C_blocks = [(1,1,1)]
        elif r_super == 6: # M7
            A_blocks = [(0,1,1), (1,1,-1)]
            B_blocks = [(1,0,1), (1,1,1)]
            C_blocks = [(0,0,1)]
            
        for r_inner in range(7):
            col = base + r_inner
            # U
            for bi, bj, coeff in A_blocks:
                # Local U2 indices: i2, j2
                for i2 in range(2):
                    for j2 in range(2):
                        gi = bi * 2 + i2
                        gj = bj * 2 + j2
                        U[gi * 4 + gj, col] += coeff * U2[i2 * 2 + j2, r_inner]
                        
            # V
            for bj, bk, coeff in B_blocks:
                for j2 in range(2):
                    for k2 in range(2):
                        gj = bj * 2 + j2
                        gk = bk * 2 + k2
                        V[gj * 4 + gk, col] += coeff * V2[j2 * 2 + k2, r_inner]
                        
            # W
            for bk, bi, coeff in C_blocks:
                for k2 in range(2):
                    for i2 in range(2):
                        gk = bk * 2 + k2
                        gi = bi * 2 + i2
                        W[gk * 4 + gi, col] += coeff * W2[k2 * 2 + i2, r_inner]
                        
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook_443():
    n, m, p = 4, 4, 3
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Try Strassen 444 + Schoolbook 443
    U1, V1, W1 = _strassen_444()
    U2, V2, W2 = _schoolbook_443()
    
    # Combine
    # U1 is 16x49, U2 is 16x48
    # V1 is 28x49, V2 is 12x48
    # W1 is 28x49, W2 is 12x48
    # Need to align W indices.
    # W1 handles k=0..3. W2 handles k=0..2 (local).
    # We need W2 to handle k=4..6.
    # Shift W2 indices by 4.
    # W2 row index is k_local * 4 + i.
    # We want gk = k_local + 4.
    # So new row index is (k_local + 4) * 4 + i = k_local * 4 + i + 16.
    
    W2_shifted = np.zeros((p * n, len(W2[0])))
    for r in range(len(W2[0])):
        for k_local in range(3):
            for i in range(4):
                W2_shifted[(k_local + 4) * 4 + i, r] = W2[k_local * 4 + i, r]
    
    # Concatenate
    U = np.hstack([np.array(U1), np.array(U2)])
    V = np.hstack([np.array(V1), np.array(V2)])
    W = np.hstack([np.array(W1), W2_shifted])
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback to schoolbook 447
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
# EVOLVE-BLOCK-END
