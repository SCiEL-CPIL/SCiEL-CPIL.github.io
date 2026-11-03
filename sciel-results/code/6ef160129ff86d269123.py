"""Rewritten program for matrix multiplication 4x4 x 4x9 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return U, V, W for M_2,2,2 with rank 7 (Strassen).
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: (a10+a11)b00 -> c10 += 1, c11 -= 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # r=2: a00(b01-b11) -> c01 += 1, c11 -= 1 (Wait, M3 = a00(b01-b11). C01 += M3, C11 -= M3? No.
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M7
    # So M3 contributes +1 to C01, +1 to C11?
    # Let's check standard Strassen:
    # C11 = M1 - M2 + M3 + M7
    # Yes.
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: a11(b10-b00) -> C10 += 1, C11 += 1?
    # M4 = a11(b10-b00)
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7 (M4 not in C11?)
    # Wait.
    # Standard:
    # C11 = M1 - M2 + M3 + M7
    # So M4 does NOT contribute to C11?
    # Let's re-verify Strassen formulas.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # So M4 contributes to C11 and C21.
    # In 0-indexed:
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    # Let's re-map my r indices to M0..M6.
    # My r=0 is M0.
    # My r=1 is M1.
    # My r=2 is M2.
    # My r=3 is M3.
    # My r=4 is M4.
    # My r=5 is M5.
    # My r=6 is M6.
    
    # Re-evaluating W factors:
    # r=0 (M0): C00(+1), C11(+1)
    # r=1 (M1): C10(+1), C11(-1)
    # r=2 (M2): C01(+1), C11(+1)
    # r=3 (M3): C00(+1), C10(+1)
    # r=4 (M4): C01(+1), C00(-1)
    # r=5 (M5): C00(-1), C01(+1)
    # r=6 (M6): C11(+1), C00(-1)
    
    # U factors (A indices):
    # M0: A00, A11
    # M1: A10, A11
    # M2: A00
    # M3: A11
    # M4: A00, A01
    # M5: A10, A00
    # M6: A01, A11
    
    # V factors (B indices):
    # M0: B00, B11
    # M1: B00
    # M2: B01, B11
    # M3: B10, B00
    # M4: B11
    # M5: B00, B01
    # M6: B10, B11
    
    # Reset and fill correctly
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0 (M0)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # r=1 (M1)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # r=2 (M2)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # r=3 (M3)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # r=4 (M4)
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[0,4]=-1
    
    # r=5 (M5)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=-1; W[1,5]=1
    
    # r=6 (M6)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[3,6]=1; W[0,6]=-1
    
    return U, V, W

def _schoolbook_4x4x1():
    """
    Return U, V, W for M_4,4,1 with rank 16.
    U: (16, 16)
    V: (4, 16)  (Input B is 4x1, so 4 entries)
    W: (4, 16)  (Output C is 4x1, so 4 entries)
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U[i * 4 + j, r] = 1.0
            V[j, r] = 1.0
            W[i, r] = 1.0
    return U, V, W

def _strassen_4x4x4():
    """
    Construct M_4,4,4 decomposition using recursive Strassen.
    Rank 49.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank444 = 49
    U4 = np.zeros((16, rank444))
    V4 = np.zeros((16, rank444))
    W4 = np.zeros((16, rank444))
    
    # Outer Strassen factors (for 2x2 blocks of 2x2 matrices)
    # Same U2, V2, W2 structure but applied to block indices.
    # Block index p,q in 0..1.
    # U_out[p, q, k] corresponds to U2[p*2+q, k]?
    # Yes, the structure is identical.
    
    for k in range(7): # Outer term
        for l in range(7): # Inner term
            col = k * 7 + l
            # Combine U
            for p in range(2):
                for q in range(2):
                    # U2 index for block (p,q) is p*2+q
                    u_out = U2[p * 2 + q, k]
                    if u_out != 0:
                        for i in range(2):
                            for j in range(2):
                                # Global A index: 2p+i, 2q+j -> (2p+i)*4 + (2q+j)
                                idx_a = (2 * p + i) * 4 + (2 * q + j)
                                # U2 index for inner (i,j) is i*2+j
                                u_in = U2[i * 2 + j, l]
                                U4[idx_a, col] += u_out * u_in
                                
            # Combine V
            for p in range(2): # Block row? No, V is for B.
                # B blocks are indexed by (p, q) where p is row of block, q is col of block.
                # In V2, the index is (j_block, k_block).
                # Let's use indices (p, q) for block position in B.
                for q in range(2):
                    v_out = V2[p * 2 + q, k]
                    if v_out != 0:
                        for i in range(2): # Inner row j
                            for j in range(2): # Inner col k
                                # Global B index: (2p+i, 2q+j) -> (2p+i)*4 + (2q+j)
                                idx_b = (2 * p + i) * 4 + (2 * q + j)
                                v_in = V2[i * 2 + j, l]
                                V4[idx_b, col] += v_out * v_in
                                
            # Combine W
            # W is for C. C blocks (p, q).
            for p in range(2):
                for q in range(2):
                    w_out = W2[p * 2 + q, k]
                    if w_out != 0:
                        for i in range(2):
                            for j in range(2):
                                # Global C index: (2p+i, 2q+j) -> (2q+j)*4 + (2p+i) ?
                                # Task: W[k*n+i, r]. n=4.
                                # C is 4x4. Index (row, col) = (i, k).
                                # W index = k*4 + i.
                                # Here, block (p,q) corresponds to C rows 2p..2p+1, cols 2q..2q+1.
                                # Inner index (i, j) corresponds to local row i, col j.
                                # Global row = 2p + i. Global col = 2q + j.
                                # W index = (2q + j) * 4 + (2p + i).
                                idx_c = (2 * q + j) * 4 + (2 * p + i)
                                w_in = W2[j * 2 + i, l] # W2 index is k*2+i -> j*2+i
                                W4[idx_c, col] += w_out * w_in
                                
    return U4, V4, W4

def _validate(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(T, expected, atol=tol, rtol=tol)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 115.0
    
    # Get components
    U4, V4, W4 = _strassen_4x4x4()
    U1, V1, W1 = _schoolbook_4x4x1()
    
    r444 = U4.shape[1] # 49
    r441 = U1.shape[1] # 16
    total_rank = 2 * r444 + r441 # 114
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: Cols 0-3
    U[:, 0:r444] = U4
    # V: B cols 0-3. Local index j*4+k. Global j*9+k.
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:r444] = V4[j * 4 + k, 0:r444]
    # W: C cols 0-3. Local index k*4+i. Global k*4+i.
    W[0:16, 0:r444] = W4[0:16, 0:r444]
    
    # Block 2: Cols 4-7
    U[:, r444:2*r444] = U4
    # V: B cols 4-7. Local index j*4+k. Global j*9+(4+k).
    for j in range(4):
        for k in range(4):
            V[j * 9 + 4 + k, r444:2*r444] = V4[j * 4 + k, 0:r444]
    # W: C cols 4-7. Local index k*4+i. Global (4+k)*4+i.
    # Global indices 16:32.
    W[16:32, r444:2*r444] = W4[0:16, 0:r444]
    
    # Block 3: Col 8
    U[:, 2*r444:2*r444+r441] = U1
    # V: B col 8. Local index j. Global j*9+8.
    for j in range(4):
        V[j * 9 + 8, 2*r444:2*r444+r441] = V1[j, 0:r441]
    # W: C col 8. Local index i. Global 8*4+i = 32+i.
    W[32:36, 2*r444:2*r444+r441] = W1[0:4, 0:r441]
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
