"""
Program to find a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
Uses block Strassen decomposition: partition into 2x2 and 1x1 sub-blocks,
apply rank-7 Strassen to 2x2x2 sub-problems and rank-1 schoolbook to 1x1x1 sub-problems.
This reduces rank from 90 (schoolbook) to 84.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0

def _construct_schoolbook():
    """
    Constructs the standard schoolbook decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with index k*n+i
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Each entry is (row_U, col_V, val_U, val_V, val_W)
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    W index: k*n + i where n=2, so W[k*2+i]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A indexing: A[i,j] -> row i*2+j
    # B indexing: B[j,k] -> row j*2+k
    # C indexing: C[i,k] -> W[k*2+i]
    
    # Each Strassen product: list of (a_row, b_row, sign)
    # where contribution to C is sum of sign * A[a_row] * B[b_row]
    
    # M1 = (A11+A22)(B11+B22)
    # A11=row0, A22=row3, B11=row0, B22=row3
    # M1 contributes: +A11*B11 + A11*B22 + A22*B11 + A22*B22
    # C11 += M1, C22 += M1
    
    # We build U, V, W columns for each of the 7 Strassen multipliers
    # For each multiplier r, U_col has entries for A rows, V_col for B rows, W_col for C rows
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # U col 0: A11=1, A22=1
    U[0][0] = 1; U[3][0] = 1
    # V col 0: B11=1, B22=1
    V[0][0] = 1; V[3][0] = 1
    # W col 0: C11=+1, C22=+1
    W[0][0] = 1  # C[0,0]: k=0,i=0 -> idx 0
    W[3][0] = 1  # C[1,1]: k=1,i=1 -> idx 3
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2
    W[2][1] = 1  # C[1,0]: k=0,i=1 -> idx 2
    
    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C12 += M3
    W[1][2] = 1  # C[0,1]: k=1,i=0 -> idx 1
    
    # M4 = A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5
    W[0][4] = -1  # C[0,0]
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[3][5] = 1  # C[1,1]
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0][6] = 1  # C[0,0]
    
    return U, V, W

def _construct_block_strassen():
    """
    Constructs block Strassen decomposition for 3x5x6.
    
    Partition:
    - A: 3x5 -> rows [0:2], [2:3]; cols [0:2], [2:5]
    - B: 5x6 -> rows [0:2], [2:5]; cols [0:2], [2:6]
    - C: 3x6 -> rows [0:2], [2:3]; cols [0:2], [2:6]
    
    Sub-blocks:
    (i_block, j_block, k_block) where:
    i_block: 0 (rows 0-1), 1 (row 2)
    j_block: 0 (cols 0-1), 1 (cols 2-4)
    k_block: 0 (cols 0-1), 1 (cols 2-5)
    
    For each (i_block, j_block, k_block):
    - If all dimensions are 2: use Strassen rank 7
    - Otherwise: use schoolbook rank = dim_i * dim_j * dim_k
    
    Sizes:
    i_block=0: size 2, i_block=1: size 1
    j_block=0: size 2, j_block=1: size 3
    k_block=0: size 2, k_block=1: size 4
    
    Sub-problem ranks:
    (0,0,0): 2x2x2 -> Strassen rank 7
    (0,0,1): 2x2x4 -> schoolbook 16
    (0,1,0): 2x3x2 -> schoolbook 12
    (0,1,1): 2x3x4 -> schoolbook 24
    (1,0,0): 1x2x2 -> schoolbook 4
    (1,0,1): 1x2x4 -> schoolbook 8
    (1,1,0): 1x3x2 -> schoolbook 6
    (1,1,1): 1x3x4 -> schoolbook 12
    
    Total: 7+16+12+24+4+8+6+12 = 89... let me recheck.
    
    Wait, I should try different partitions. Let me think about this more carefully.
    
    Actually, for 3x5x6, let's try:
    Split A rows into [2,1], A cols into [2,3]
    Split B rows into [2,3], B cols into [2,4]
    Split C rows into [2,1], C cols into [2,4]
    
    Sub-blocks (i_sz, j_sz, k_sz):
    (2,2,2): Strassen 7
    (2,2,4): schoolbook 16
    (2,3,2): schoolbook 12
    (2,3,4): schoolbook 24
    (1,2,2): schoolbook 4
    (1,2,4): schoolbook 8
    (1,3,2): schoolbook 6
    (1,3,4): schoolbook 12
    Total: 7+16+12+24+4+8+6+12 = 89
    
    Hmm, that's only 1 less than 90. Let me try other splits.
    
    What if we split differently? Try A cols [3,2], B rows [3,2]:
    (2,3,2): 12, (2,3,4): 24, (2,2,2): 7, (2,2,4): 16
    (1,3,2): 6, (1,3,4): 12, (1,2,2): 4, (1,2,4): 8
    Same total: 89.
    
    What about splitting B cols as [4,2] instead of [2,4]?
    (2,2,4): 16, (2,2,2): 7, (2,3,4): 24, (2,3,2): 12
    (1,2,4): 8, (1,2,2): 4, (1,3,4): 12, (1,3,2): 6
    Same total: 89.
    
    What if we use Strassen on more blocks? The only 2x2x2 block gets Strassen.
    
    Actually wait - can we split to get more 2x2x2 blocks? 
    n=3: [2,1] or [1,2]
    m=5: [2,3] or [3,2] or [2,2,1] etc.
    p=6: [2,4] or [4,2] or [2,2,2] etc.
    
    If p=6 split as [2,2,2]:
    i=[2,1], m=[2,3], p=[2,2,2]:
    (2,2,2): 7, (2,2,2): 7, (2,2,2): 7
    (2,3,2): 12, (2,3,2): 12, (2,3,2): 12
    (1,2,2): 4, (1,2,2): 4, (1,2,2): 4
    (1,3,2): 6, (1,3,2): 6, (1,3,2): 6
    Total: 21+36+12+18 = 87
    
    Even better! What about m=[3,2]?
    Same by symmetry: 87.
    
    What about i=[1,2], m=[2,3], p=[2,2,2]?
    (1,2,2): 4, (1,2,2): 4, (1,2,2): 4
    (1,3,2): 6, (1,3,2): 6, (1,3,2): 6
    (2,2,2): 7, (2,2,2): 7, (2,2,2): 7
    (2,3,2): 12, (2,3,2): 12, (2,3,2): 12
    Total: 12+18+21+36 = 87. Same.
    
    Can we do even better? What about m=[2,2,1]?
    i=[2,1], m=[2,2,1], p=[2,2,2]:
    (2,2,2): 7 x3 = 21
    (2,2,2): 7 x3 = 21
    (2,1,2): 4 x3 = 12
    (1,2,2): 4 x3 = 12
    (1,2,2): 4 x3 = 12
    (1,1,2): 2 x3 = 6
    Total: 21+21+12+12+12+6 = 84!
    
    Yes! 84 is the target.
    """
    n, m, p = DIMENSIONS
    
    # Partition:
    # A rows: [0:2], [2:3] -> sizes [2, 1]
    # A cols: [0:2], [2:4], [4:5] -> sizes [2, 2, 1]
    # B rows: [0:2], [2:4], [4:5] -> sizes [2, 2, 1]
    # B cols: [0:2], [2:4], [4:6] -> sizes [2, 2, 2]
    # C rows: [0:2], [2:3] -> sizes [2, 1]
    # C cols: [0:2], [2:4], [4:6] -> sizes [2, 2, 2]
    
    i_splits = [0, 2, 3]  # A row boundaries
    j_splits = [0, 2, 4, 5]  # A col / B row boundaries
    k_splits = [0, 2, 4, 6]  # B col / C col boundaries
    
    i_sizes = [i_splits[i+1] - i_splits[i] for i in range(len(i_splits)-1)]
    j_sizes = [j_splits[j+1] - j_splits[j] for j in range(len(j_splits)-1)]
    k_sizes = [k_splits[k+1] - k_splits[k] for k in range(len(k_splits)-1)]
    
    # i_sizes = [2, 1], j_sizes = [2, 2, 1], k_sizes = [2, 2, 2]
    
    total_rank = 0
    for ii in range(len(i_sizes)):
        for jj in range(len(j_sizes)):
            for kk in range(len(k_sizes)):
                si, sj, sk = i_sizes[ii], j_sizes[jj], k_sizes[kk]
                if si == 2 and sj == 2 and sk == 2:
                    total_rank += 7
                else:
                    total_rank += si * sj * sk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ii in range(len(i_sizes)):
        i_start = i_splits[ii]
        i_end = i_splits[ii+1]
        i_sz = i_sizes[ii]
        for jj in range(len(j_sizes)):
            j_start = j_splits[jj]
            j_end = j_splits[jj+1]
            j_sz = j_sizes[jj]
            for kk in range(len(k_sizes)):
                k_start = k_splits[kk]
                k_end = k_splits[kk+1]
                k_sz = k_sizes[kk]
                
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    # Use Strassen rank-7
                    sU, sV, sW = _strassen_2x2x2()
                    for r in range(7):
                        for a_row in range(4):
                            if sU[a_row][r] != 0:
                                # A local row a_row -> global row i_start + a_row//2, col j_start + a_row%2
                                gi = i_start + a_row // 2
                                gj = j_start + a_row % 2
                                U[gi * m + gj][col] = sU[a_row][r]
                        for b_row in range(4):
                            if sV[b_row][r] != 0:
                                # B local row b_row -> global row j_start + b_row//2, col k_start + b_row%2
                                gj = j_start + b_row // 2
                                gk = k_start + b_row % 2
                                V[gj * p + gk][col] = sV[b_row][r]
                        for c_row in range(4):
                            if sW[c_row][r] != 0:
                                # C local: c_row = k_local * 2 + i_local
                                ki = c_row // 2
                                ii_local = c_row % 2
                                gi = i_start + ii_local
                                gk = k_start + ki
                                W[gk * n + gi][col] = sW[c_row][r]
                        col += 1
                else:
                    # Schoolbook
                    for a_i in range(i_sz):
                        for a_j in range(j_sz):
                            for b_k in range(k_sz):
                                gi = i_start + a_i
                                gj = j_start + a_j
                                gk = k_start + b_k
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=3, m=5, p=6):
    """
    Verify that U, V, W correctly decompose matrix multiplication.
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    For all other (i,j,k) combinations where the indices don't form a valid path, it should be 0.
    Actually, the tensor T[i,j,k] = 1 if the indices are compatible (always true for valid matmul),
    and the decomposition requires: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(np.multiply(u_row, v_row), w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Entry point to return the factor matrices.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Try block Strassen first
    try:
        result = _construct_block_strassen()
        if _verify_decomposition(result):
            if time.time() < deadline:
                return result
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = _construct_schoolbook()
    return result

# EVOLVE-BLOCK-END
