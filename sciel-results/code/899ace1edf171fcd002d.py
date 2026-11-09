"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c] for A[i,j], V[i*2+j, c] for B[i,j], W[i*2+j, c] for C[i,j]
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U rows: A00, A01, A10, A11 (indices 0,1,2,3)
    # V rows: B00, B01, B10, B11 (indices 0,1,2,3)
    # W rows: C00, C01, C10, C11 (indices 0,1,2,3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # M3: A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    
    # M4: A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    
    # M5: (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[2, 5] = 1  # B10
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[1, 6] = 1  # B01
    V[3, 6] = 1  # B11
    
    # W: reconstruct C from M1..M7
    # C00 = M1+M4-M5+M7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C01 = M3+M5
    W[1, 2] = 1
    W[1, 4] = 1
    
    # C10 = M2+M4
    W[2, 1] = 1
    W[2, 3] = 1
    
    # C11 = M1-M2+M3+M6
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p, offset=0):
    """Return U, V, W for schoolbook n x m x p block.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
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

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Split inner dimension m=5 into 1+2+2
    # A: 2x5 = [A1 (2x1) | A2 (2x2) | A3 (2x2)]
    # B: 5x6 = [B1 (1x6); B2 (2x6); B3 (2x6)]
    # C = A1@B1 + A2@B2 + A3@B3
    #
    # A1@B1: 2x1x6, schoolbook rank 12
    # A2@B2: 2x2x6, split into three 2x2x2 Strassen blocks, rank 21
    # A3@B3: 2x2x6, split into three 2x2x2 Strassen blocks, rank 21
    #
    # Total rank: 12 + 21 + 21 = 54
    
    total_rank = 12 + 21 + 21  # 54
    
    U_full = np.zeros((10, total_rank))
    V_full = np.zeros((30, total_rank))
    W_full = np.zeros((12, total_rank))
    
    col_offset = 0
    
    # Part 1: A1@B1 where A1 is 2x1, B1 is 1x6
    # Schoolbook: rank = 2*1*6 = 12
    # A[i,j] -> U row i*5+j. A1 uses j=0, so U rows 0 and 5.
    # B[j,k] -> V row j*6+k. B1 uses j=0, so V rows 0..5.
    # C[i,k] -> W row k*2+i.
    
    for i in range(2):
        for k in range(6):
            U_full[i * 5, col_offset] = 1       # A[i,0]
            V_full[k, col_offset] = 1            # B[0,k]
            W_full[k * 2 + i, col_offset] = 1    # C[i,k]
            col_offset += 1
    
    # Part 2: A2@B2 where A2 is 2x2 (cols 1,2), B2 is 2x6 (rows 1,2)
    # A2: A[0,1], A[0,2], A[1,1], A[1,2] -> U rows 1,2,3,4
    # B2: B[1,0..5], B[2,0..5] -> V rows 6..17
    # Split into three 2x2x2 Strassen blocks
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for b in range(3):
        # Block b handles B2 columns 2*b to 2*b+1
        # A2 rows: A[0,1], A[0,2], A[1,1], A[1,2] -> U rows 1,2,6,7
        # Strassen U rows: 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1]
        # Map: U_str[0]->U_full[1], U_str[1]->U_full[2], U_str[2]->U_full[6], U_str[3]->U_full[7]
        U_full[1, col_offset:col_offset+7] = U_str[0, :]
        U_full[2, col_offset:col_offset+7] = U_str[1, :]
        U_full[6, col_offset:col_offset+7] = U_str[2, :]
        U_full[7, col_offset:col_offset+7] = U_str[3, :]
        
        # B2 block rows:
        # B[1,2*b], B[1,2*b+1], B[2,2*b], B[2,2*b+1]
        # V row for B[j,k] = j*6+k
        V_full[6+2*b, col_offset:col_offset+7] = V_str[0, :]      # B[1,2*b]
        V_full[6+2*b+1, col_offset:col_offset+7] = V_str[1, :]    # B[1,2*b+1]
        V_full[12+2*b, col_offset:col_offset+7] = V_str[2, :]     # B[2,2*b]
        V_full[12+2*b+1, col_offset:col_offset+7] = V_str[3, :]   # B[2,2*b+1]
        
        # C rows: C[0,2*b], C[0,2*b+1], C[1,2*b], C[1,2*b+1]
        W_full[4*b, col_offset:col_offset+7] = W_str[0, :]
        W_full[4*b+2, col_offset:col_offset+7] = W_str[1, :]
        W_full[4*b+1, col_offset:col_offset+7] = W_str[2, :]
        W_full[4*b+3, col_offset:col_offset+7] = W_str[3, :]
        
        col_offset += 7
    
    # Part 3: A3@B3 where A3 is 2x2 (cols 3,4), B3 is 2x6 (rows 3,4)
    # A3: A[0,3], A[0,4], A[1,3], A[1,4] -> U rows 5,6,7,8,9... wait
    # A[i,j] -> U row i*5+j
    # A[0,3] -> U row 3, A[0,4] -> U row 4, A[1,3] -> U row 8, A[1,4] -> U row 9
    
    for b in range(3):
        # A3 rows: A[0,3], A[0,4], A[1,3], A[1,4] -> U rows 3,4,8,9
        # But Strassen U expects rows in order: A[0,0], A[0,1], A[1,0], A[1,1]
        # So we need to map: U_str row 0 -> U_full row 3 (A[0,3])
        #                   U_str row 1 -> U_full row 4 (A[0,4])
        #                   U_str row 2 -> U_full row 8 (A[1,3])
        #                   U_str row 3 -> U_full row 9 (A[1,4])
        U_full[3, col_offset:col_offset+7] = U_str[0, :]
        U_full[4, col_offset:col_offset+7] = U_str[1, :]
        U_full[8, col_offset:col_offset+7] = U_str[2, :]
        U_full[9, col_offset:col_offset+7] = U_str[3, :]
        
        # B3 block rows:
        # B[3,2*b], B[3,2*b+1], B[4,2*b], B[4,2*b+1]
        # V row for B[j,k] = j*6+k
        V_full[18+2*b, col_offset:col_offset+7] = V_str[0, :]      # B[3,2*b]
        V_full[18+2*b+1, col_offset:col_offset+7] = V_str[1, :]    # B[3,2*b+1]
        V_full[24+2*b, col_offset:col_offset+7] = V_str[2, :]      # B[4,2*b]
        V_full[24+2*b+1, col_offset:col_offset+7] = V_str[3, :]    # B[4,2*b+1]
        
        # C rows: same as before
        W_full[4*b, col_offset:col_offset+7] = W_str[0, :]
        W_full[4*b+2, col_offset:col_offset+7] = W_str[1, :]
        W_full[4*b+1, col_offset:col_offset+7] = W_str[2, :]
        W_full[4*b+3, col_offset:col_offset+7] = W_str[3, :]
        
        col_offset += 7
    
    # Verify
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
