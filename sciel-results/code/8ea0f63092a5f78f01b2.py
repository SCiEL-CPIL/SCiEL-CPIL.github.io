import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W decompose the tensor T[i,j,k]=1.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    R = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        if not np.allclose(result, 1.0, atol=tol, rtol=tol):
            return False
    except Exception:
        return False
    return True

def _round_to_half_integer(arr, tol=1e-9):
    """
    Round entries to nearest half-integer.
    """
    scaled = 2.0 * np.array(arr, dtype=np.float64)
    rounded = np.round(scaled)
    return rounded / 2.0

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (Rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    # Standard Strassen decomposition
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A is 2x2: [[A11, A12], [A21, A22]]
    # B is 2x2: [[B11, B12], [B21, B22]]
    # C = AB: [[C11, C12], [C21, C22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U factors A:
    # U[i*2+j, c] corresponds to coefficient of A_ij in M_c
    # c=0: M1: A11, A22 -> U[0,0]=1, U[3,0]=1
    # c=1: M2: A21, A22 -> U[2,1]=1, U[3,1]=1
    # c=2: M3: A11 -> U[0,2]=1
    # c=3: M4: A22 -> U[3,3]=1
    # c=4: M5: A11, A12 -> U[0,4]=1, U[1,4]=1
    # c=5: M6: -A11, A21 -> U[0,5]=-1, U[2,5]=1
    # c=6: M7: A12, -A22 -> U[1,6]=1, U[3,6]=-1
    
    U = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[0][5] = -1; U[2][5] = 1
    U[1][6] = 1; U[3][6] = -1
    
    # V factors B:
    # c=0: M1: B12, B21 -> V[1,0]=1, V[2,0]=1
    # c=1: M2: B11 -> V[0,1]=1
    # c=2: M3: B11, -B21 -> V[0,2]=1, V[2,2]=-1
    # c=3: M4: -B12, B22 -> V[1,3]=-1, V[3,3]=1
    # c=4: M5: B22 -> V[3,4]=1
    # c=5: M6: B11, B12 -> V[0,5]=1, V[1,5]=1
    # c=6: M7: B21, B22 -> V[2,6]=1, V[3,6]=1
    
    V = [[0]*7 for _ in range(4)]
    V[1][0] = 1; V[2][0] = 1
    V[0][1] = 1
    V[0][2] = 1; V[2][2] = -1
    V[1][3] = -1; V[3][3] = 1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    # W factors C:
    # C11 (k=0,i=0 -> W[0]): M1, M4, -M5, M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    # C12 (k=1,i=0 -> W[1]): M3, M5 -> W[1,2]=1, W[1,4]=1
    # C21 (k=0,i=1 -> W[2]): M2, M4 -> W[2,1]=1, W[2,3]=1
    # C22 (k=1,i=1 -> W[3]): M1, -M2, M3, M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    
    W = [[0]*7 for _ in range(4)]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _block_decompose_2x4x7():
    """
    Decomposes 2x4x7 using block structure.
    A (2x4) = [A1, A2] where A1, A2 are 2x2.
    B (4x7) = [B1; B2] where B1, B2 are 2x7.
    C = A B = A1 B1 + A2 B2.
    
    We decompose A1 B1 (2x2x7) and A2 B2 (2x2x7) separately.
    For 2x2x7, we can view it as 2x2x(2+2+2+1).
    Actually, simpler: 2x2x7 is a sub-tensor.
    We can use Strassen for the 2x2x2 parts and schoolbook for the rest?
    Or just use Strassen on 2x2x2 blocks and combine.
    
    Let's break B (4x7) into B11, B12, B13, B14 (each 2x2, except last is 2x1? No, 7 is odd).
    B is 4x7.
    Rows 0-1: B_top (2x7)
    Rows 2-3: B_bot (2x7)
    
    C = A_top * B_top + A_bot * B_bot? No.
    A is 2x4.
    A = [A_col1 A_col2] where A_col1 is 2x2, A_col2 is 2x2.
    B is 4x7.
    B = [B_row1; B_row2] where B_row1 is 2x7, B_row2 is 2x7.
    
    C = A_col1 * B_row1 + A_col2 * B_row2.
    
    So we need to multiply (2x2) * (2x7) twice and add.
    Let's find a decomposition for M(2,2,7).
    
    M(2,2,7) can be seen as:
    A (2x2), B (2x7), C (2x7).
    C = A B.
    
    We can decompose B into four 2x2 blocks and one 2x1 block?
    B = [B0 B1 B2 B3] where B0..B3 are 2x2, and... wait 2x7.
    B = [B0 B1 B2 B3] where B0, B1, B2 are 2x2 and B3 is 2x1.
    Then C = [C0 C1 C2 C3] where C0,C1,C2 are 2x2 and C3 is 2x1.
    C0 = A B0, C1 = A B1, C2 = A B2, C3 = A B3.
    
    Rank(M(2,2,2)) = 7 (Strassen).
    Rank(M(2,2,1)):
    A (2x2) * B (2x1) = C (2x1).
    C1 = A11 B1 + A12 B2
    C2 = A21 B1 + A22 B2
    This is bilinear in A and B.
    Variables: A11, A12, A21, A22, B1, B2. Output C1, C2.
    Tensor T[a1,a2,b1,b2] -> c1, c2.
    T is shape (2,2,2,2,2,2) -> (2,2).
    Actually, let's just use schoolbook for 2x2x1. Rank = 4.
    Or can we do better?
    C1 = A11 B1 + A12 B2
    C2 = A21 B1 + A22 B2
    This is essentially 2 independent 2x1 multiplications? No.
    It's a bilinear form.
    Let's check rank.
    The tensor is T[i, j, k, l] where i,j index A, k,l index B?
    A is 2x2, B is 2x1.
    Indices for A: (0,0), (0,1), (1,0), (1,1).
    Indices for B: (0,0), (1,0).
    Output C: (0,0), (1,0).
    
    Let's use schoolbook rank 4 for 2x2x1.
    So Rank(M(2,2,7)) <= 3 * 7 + 4 = 25.
    
    Total rank for 2x4x7 = 2 * Rank(M(2,2,7)) = 2 * 25 = 50.
    This matches the "knowledge" hint!
    """
    
    U_222, V_222, W_222 = _strassen_2x2x2()
    
    # Schoolbook for 2x2x1
    # A (2x2), B (2x1), C (2x1)
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # Rank 4 decomposition:
    # c=0: A[0,0]B[0,0] -> C[0]
    # c=1: A[0,1]B[1,0] -> C[0]
    # c=2: A[1,0]B[0,0] -> C[1]
    # c=3: A[1,1]B[1,0] -> C[1]
    
    U_221 = [[0]*4 for _ in range(4)]
    V_221 = [[0]*4 for _ in range(2)]
    W_221 = [[0]*4 for _ in range(2)]
    
    U_221[0][0] = 1 # A00
    V_221[0][0] = 1 # B00
    W_221[0][0] = 1 # C0
    
    U_221[1][1] = 1 # A01
    V_221[1][1] = 1 # B10
    W_221[0][1] = 1 # C0
    
    U_221[2][2] = 1 # A10
    V_221[0][2] = 1 # B00
    W_221[1][2] = 1 # C1
    
    U_221[3][3] = 1 # A11
    V_221[1][3] = 1 # B10
    W_221[1][3] = 1 # C1
    
    # Now assemble 2x4x7
    # A (2x4) = [A1 A2]
    # B (4x7) = [B1; B2]
    # C = A1 B1 + A2 B2
    
    # B1 (2x7) = [B1_0 B1_1 B1_2 B1_3] where B1_0..2 are 2x2, B1_3 is 2x1
    # B2 (2x7) = [B2_0 B2_1 B2_2 B2_3] where B2_0..2 are 2x2, B2_3 is 2x1
    
    # We need to construct U, V, W for the whole thing.
    # U will have shape (8, R_total)
    # V will have shape (28, R_total)
    # W will have shape (14, R_total)
    
    # R_total = 2 * (3*7 + 4) = 2 * 25 = 50
    
    R_total = 50
    
    U = [[0]*R_total for _ in range(8)]
    V = [[0]*R_total for _ in range(28)]
    W = [[0]*R_total for _ in range(14)]
    
    # Helper to place a block
    def place_block(U_block, V_block, W_block, offset, 
                    u_row_start, v_row_start, w_row_start):
        """
        Place block matrices into the global matrices.
        u_row_start: starting row in U for this block (0 for A1, 4 for A2)
        v_row_start: starting row in V for this block (0 for B1, 14 for B2? No, B1 is 2x7 -> 14 rows. B2 is 2x7 -> 14 rows. Total 28.)
        w_row_start: starting row in W for this block (0 for C from A1B1? No, W is for C which is 2x7. 
                     Wait, C is the sum. So W factors must add up.
                     W for A1B1 contributes to C. W for A2B2 contributes to C.
                     So they share the same W indices? No, the rank is the sum of ranks.
                     The decomposition is a direct sum.
                     So W_block for A1B1 is placed at w_row_start.
                     W_block for A2B2 is placed at w_row_start + R_block?
                     No, the output C is the same C.
                     C = C1 + C2.
                     The tensor is T = T1 + T2.
                     The decomposition is (U1, V1, W1) for T1 and (U2, V2, W2) for T2.
                     The combined decomposition is (U1 || U2, V1 || V2, W1 || W2) ?
                     No.
                     T1[i,j,k] = sum_c U1[i,j,c] V1[j,k,c] W1[k,i,c]
                     T2[i,j,k] = sum_d U2[i,j,d] V2[j,k,d] W2[k,i,d]
                     T[i,j,k] = T1 + T2 = sum_c ... + sum_d ...
                     So we can just concatenate the columns of U, V, W.
                     U = [U1 U2], V = [V1 V2], W = [W1 W2].
                     The rank is R1 + R2.
                     And the indices for U, V, W in the global arrays are just the local indices.
                     So w_row_start is always 0?
                     Yes, because W1 and W2 are added.
                     So W_global[k*n+i, c] = W1[k*n+i, c] + W2[k*n+i, c]?
                     No, if we concatenate columns, W_global has R1+R2 columns.
                     W_global[:, 0:R1] = W1
                     W_global[:, R1:R1+R2] = W2
                     Then sum_c U_global V_global W_global = sum_{c in R1} ... + sum_{d in R2} ...
                     This works.
                     So w_row_start is not needed for W in the sense of offset, but the column offset is.
                     However, U and V also have column offsets.
                     U_global[:, 0:R1] = U1
                     U_global[:, R1:R1+R2] = U2
                     
                     So we just need column offsets.
                     u_row_start, v_row_start, w_row_start are for the ROWS (matrix dimensions).
                     The COLUMNS are offset by the cumulative rank.
        """
        pass
    
    # Let's do it manually with column offsets.
    
    # Block 1: A1 (2x2) * B1 (2x7)
    # A1 is rows 0-3 of U (indices 0,1,2,3)
    # B1 is rows 0-13 of V (indices 0..13)
    # C is rows 0-13 of W (indices 0..13)
    
    # B1 decomposes into B1_0 (2x2), B1_1 (2x2), B1_2 (2x2), B1_3 (2x1)
    # V indices for B1:
    # B1_0: V[0:2, :] (rows 0,1)
    # B1_1: V[2:4, :] (rows 2,3)
    # B1_2: V[4:6, :] (rows 4,5)
    # B1_3: V[6:8, :] (rows 6,7) -- Wait, B1 is 2x7.
    # B1 = [B1_0 B1_1 B1_2 B1_3]
    # B1_0 is cols 0-1, B1_1 cols 2-3, B1_2 cols 4-5, B1_3 cols 6.
    # V index = j*p + k. j is 0 or 1. k is 0..6.
    # B1_0: j=0,1; k=0,1. V indices: 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8.
    # B1_1: j=0,1; k=2,3. V indices: 2,3,9,10.
    # B1_2: j=0,1; k=4,5. V indices: 4,5,11,12.
    # B1_3: j=0,1; k=6. V indices: 6,13.
    
    # W indices for C:
    # C is 2x7. W index = k*n + i. k=0..6, i=0,1.
    # C_0 (2x2): k=0,1; i=0,1. W indices: 0,1,2,3.
    # C_1 (2x2): k=2,3; i=0,1. W indices: 4,5,6,7.
    # C_2 (2x2): k=4,5; i=0,1. W indices: 8,9,10,11.
    # C_3 (2x1): k=6; i=0,1. W indices: 12,13.
    
    # Let's define a helper to add a sub-decomposition.
    
    def add_sub(U_sub, V_sub, W_sub, col_offset, 
                u_rows, v_rows, w_rows):
        """
        U_sub: (4, r_sub) for 2x2x2 or (4, r_sub) for 2x2x1
        V_sub: (4, r_sub) or (2, r_sub)
        W_sub: (4, r_sub) or (2, r_sub)
        
        u_rows: list of 4 indices in U
        v_rows: list of 4 indices in V (or 2)
        w_rows: list of 4 indices in W (or 2)
        """
        r_sub = len(U_sub[0])
        for i_sub in range(len(U_sub)):
            for c in range(r_sub):
                U[u_rows[i_sub]][col_offset + c] += U_sub[i_sub][c]
        for j_sub in range(len(V_sub)):
            for c in range(r_sub):
                V[v_rows[j_sub]][col_offset + c] += V_sub[j_sub][c]
        for k_sub in range(len(W_sub)):
            for c in range(r_sub):
                W[w_rows[k_sub]][col_offset + c] += W_sub[k_sub][c]
        return r_sub
        
    # Block 1: A1 * B1
    # A1 rows in U: [0, 1, 2, 3] (A11, A12, A21, A22)
    # B1_0 rows in V: [0, 1, 7, 8]
    # C_0 rows in W: [0, 1, 2, 3]
    
    col_off = 0
    col_off += add_sub(U_222, V_222, W_222, col_off, 
                       [0, 1, 2, 3], [0, 1, 7, 8], [0, 1, 2, 3])
    
    # B1_1 rows in V: [2, 3, 9, 10]
    # C_1 rows in W: [4, 5, 6, 7]
    col_off += add_sub(U_222, V_222, W_222, col_off, 
                       [0, 1, 2, 3], [2, 3, 9, 10], [4, 5, 6, 7])
    
    # B1_2 rows in V: [4, 5, 11, 12]
    # C_2 rows in W: [8, 9, 10, 11]
    col_off += add_sub(U_222, V_222, W_222, col_off, 
                       [0, 1, 2, 3], [4, 5, 11, 12], [8, 9, 10, 11])
    
    # B1_3 rows in V: [6, 13]
    # C_3 rows in W: [12, 13]
    col_off += add_sub(U_221, V_221, W_221, col_off, 
                       [0, 1, 2, 3], [6, 13], [12, 13])
    
    # Block 2: A2 * B2
    # A2 rows in U: [4, 5, 6, 7]
    # B2 is rows 14-27 of V.
    # B2_0: j=0,1; k=0,1. V indices: 14, 15, 21, 22.
    # B2_1: j=0,1; k=2,3. V indices: 16, 17, 23, 24.
    # B2_2: j=0,1; k=4,5. V indices: 18, 19, 25, 26.
    # B2_3: j=0,1; k=6. V indices: 20, 27.
    
    # C is same W indices.
    
    col_off += add_sub(U_222, V_222, W_222, col_off, 
                       [4, 5, 6, 7], [14, 15, 21, 22], [0, 1, 2, 3])
    
    col_off += add_sub(U_222, V_222, W_222, col_off, 
                       [4, 5, 6, 7], [16, 17, 23, 24], [4, 5, 6, 7])
    
    col_off += add_sub(U_222, V_222, W_222, col_off, 
                       [4, 5, 6, 7], [18, 19, 25, 26], [8, 9, 10, 11])
    
    col_off += add_sub(U_221, V_221, W_221, col_off, 
                       [4, 5, 6, 7], [20, 27], [12, 13])
    
    return U, V, W

def _als_decompose_explicit(n, m, p, rank, max_iter=500, tol=1e-8, seed=42, time_limit=None, start_time=None):
    """
    Explicit ALS for tensor decomposition.
    T[i,j,k] = 1 for all i,j,k.
    """
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    for it in range(max_iter):
        if time_limit and start_time and (time.time() - start_time > time_limit):
            break
            
        for i in range(n):
            for j in range(m):
                A = np.zeros((rank, p), dtype=np.float64)
                for k in range(p):
                    A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, T[i, j, :].T, rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                B = np.zeros((rank, n), dtype=np.float64)
                for i in range(n):
                    B[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(B, T[:, j, k].T, rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                C = np.zeros((rank, m), dtype=np.float64)
                for j in range(m):
                    C[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(C, T[i, :, k].T, rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        
        if it % 5 == 0:
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
            err = np.linalg.norm(recon - T)
            if err < tol:
                break
                
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    err = np.linalg.norm(recon - T)
    
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0
    
    # 1. Block Decomposition (Rank 50)
    U_blk, V_blk, W_blk = _block_decompose_2x4x7()
    
    best_result = {"U": U_blk, "V": V_blk, "W": W_blk}
    best_rank = 50
    
    # Verify block decomposition
    if _verify(U_blk, V_blk, W_blk, n, m, p, tol=1e-9):
        # It should be exact integers
        pass
    else:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = schoolbook_rank

    # 2. Search for lower rank using ALS
    # Lower bound is max(n*m, m*p, p*n) = 28
    # Try ranks from 49 down to 28
    
    for rank in range(best_rank - 1, 27, -1):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        found = False
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            seed = rank * 100 + restart
            res = _als_decompose_explicit(n, m, p, rank, max_iter=200, tol=1e-8, seed=seed, time_limit=time_limit - (time.time() - start_time), start_time=start_time)
            
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    
                    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = rank
                        found = True
                        break
        
    return best_result
