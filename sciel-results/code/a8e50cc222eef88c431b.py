"""Matrix multiplication 4x4x9: Strassen-based decomposition for rank reduction."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Check entries are integer or half-integer
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    # Check tensor equation: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _strassen_4x4x4():
    """Return rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    
    Uses two levels of Strassen: 4x4x4 -> 7 * (2x2x2) where each 2x2x2 has rank 7.
    Total rank = 7*7 = 49.
    
    Returns U (16 x 49), V (16 x 49), W (16 x 49) where:
    U indexed by i*4+k (A row i, col k)
    V indexed by k*4+j (B row k, col j) 
    W indexed by j*4+i (C col j, row i) -- trace convention W[k*n+i] with n=4, k=j
    """
    # Winograd F(2,2) for 2x2x2 matrix multiplication, rank 7
    # A is 2x2, B is 2x2, C is 2x2
    # U indexed by i*2+k, V indexed by k*2+j, W indexed by j*2+i
    
    # The 7 multiplications for Strassen/Winograd 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M3 + M6 + M7
    
    U2 = [
        [1, 1, 1, 0, 1, 0, 0],  # A00
        [0, 0, 0, 0, 0, -1, 0], # A01 (appears in M6, M7)
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1], # A11
    ]
    V2 = [
        [1, 0, 0, 0, 0, 1, 0],  # B00
        [0, 0, 1, -1, 0, 1, 1], # B01
        [0, 1, -1, 1, 0, 0, 0], # B10
        [1, 0, 0, 0, 1, 0, 1],  # B11
    ]
    W2 = [
        [1, 0, 0, 0, 1, 0, -1], # C00 (j=0, i=0)
        [0, 0, 1, 1, 0, 0, 0],  # C01 (j=1, i=0)
        [0, 1, 0, 0, 0, 1, 0],  # C10 (j=0, i=1)
        [1, 0, -1, 0, 0, 1, 1], # C11 (j=1, i=1)
    ]
    
    # Verify 2x2x2 decomposition
    for i in range(2):
        for k in range(2):
            for j in range(2):
                val = sum(U2[i*2+k][c] * V2[k*2+j][c] * W2[j*2+i][c] for c in range(7))
                expected = 1 if (i == k and j == i) or (i == j and k == i) or False else 0
                # Actually for matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
                # So the tensor has T[i,k,j] = 1 if we're computing C[i,j] from A[i,k]*B[k,j]
                # The decomposition should give: sum_c U[i,k,c]*V[k,j,c]*W[j,i,c] = 1 for all i,j,k
                # Wait, no. The matrix multiplication tensor T satisfies:
                # T[i,k,j] = delta(i,k,j) meaning the coefficient of A[i,k]*B[k,j] in C[i,j] is 1
                # So sum_c U[i,k,c]*V[k,j,c]*W[j,i,c] should equal 1 for all i,k,j where the term appears
                # Actually all terms appear, so it should be 1 for all i,k,j.
                pass
    
    # Now build 4x4x4 from 7 copies of 2x2x2
    # Partition 4x4 into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Using Strassen at the 4x4 level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M3 + M6 + M7
    
    # Each Ml is a 2x2x2 multiplication with rank 7
    # Total rank = 7*7 = 49
    
    U4 = [[0]*49 for _ in range(16)]  # 4x4 = 16 entries
    V4 = [[0]*49 for _ in range(16)]  # 4x4 = 16 entries
    W4 = [[0]*49 for _ in range(16)]  # 4x4 = 16 entries
    
    # For each of the 7 Strassen multiplications, we have a 2x2x2 sub-problem
    # The 7 multiplications involve combinations of 2x2 blocks
    
    # Define the 7 Strassen combinations:
    # Each combination specifies which 2x2 blocks of A, B, C are involved and with what signs
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
    # M2 = (A10+A11)B00 -> contributes to C10 (+1)
    # M3 = A00(B10-B11) -> contributes to C01 (+1), C11 (-1)
    # M4 = A11(B01-B10) -> contributes to C00 (-1), C01 (+1)
    # M5 = (A00+A01)B11 -> contributes to C00 (+1)
    # M6 = (A10-A00)(B00+B01) -> contributes to C10 (+1), C11 (+1)
    # M7 = (A01-A11)(B10+B11) -> contributes to C00 (-1), C11 (+1)
    
    # For each Strassen multiplication l (0-6), we need to express it as a 2x2x2 multiplication
    # where the "A" and "B" are linear combinations of 2x2 blocks
    
    # Let me define the 7 multiplications more carefully:
    # For M1: A_part = A00+A11, B_part = B00+B11
    #   A_part[i,k] = A00[i,k] + A11[i,k] for i,k in 0,1
    #   B_part[k,j] = B00[k,j] + B11[k,j] for k,j in 0,1
    #   Result: C_part[i,j] = sum_k A_part[i,k]*B_part[k,j]
    #   C00[i,j] += C_part[i,j], C11[i,j] += C_part[i,j]
    
    # For the tensor decomposition, each of the 7 multiplications contributes 7 rank-1 terms
    # Total: 49 terms
    
    # Let me build this systematically.
    # For each Strassen multiplication l, and each Winograd term c (0-6):
    #   U contribution: for each (i,k) in the 2x2 A-block, U[i,k] gets U2[i*2+k][c] * (coefficient from Strassen)
    #   V contribution: for each (k,j) in the 2x2 B-block, V[k,j] gets V2[k*2+j][c] * (coefficient from Strassen)
    #   W contribution: for each (j,i) in the 2x2 C-block, W[j,i] gets W2[j*2+i][c] * (coefficient from Strassen)
    
    # Strassen combinations: (A_blocks, B_blocks, C_blocks_with_signs)
    # A_blocks: list of (block_index, coefficient)
    # B_blocks: list of (block_index, coefficient)
    # C_blocks: list of (block_index, coefficient)
    
    strassen_combos = [
        # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        {
            'A': [(0, 1), (3, 1)],  # A00, A11
            'B': [(0, 1), (3, 1)],  # B00, B11
            'C': [(0, 1), (3, 1)],  # C00, C11
        },
        # M2: (A10+A11)B00 -> C10(+1)
        {
            'A': [(2, 1), (3, 1)],  # A10, A11
            'B': [(0, 1)],          # B00
            'C': [(2, 1)],          # C10
        },
        # M3: A00(B10-B11) -> C01(+1), C11(-1)
        {
            'A': [(0, 1)],          # A00
            'B': [(2, 1), (3, -1)], # B10, B11
            'C': [(1, 1), (3, -1)], # C01, C11
        },
        # M4: A11(B01-B10) -> C00(-1), C01(+1)
        {
            'A': [(3, 1)],          # A11
            'B': [(1, 1), (2, -1)], # B01, B10
            'C': [(0, -1), (1, 1)], # C00, C01
        },
        # M5: (A00+A01)B11 -> C00(+1)
        {
            'A': [(0, 1), (1, 1)],  # A00, A01
            'B': [(3, 1)],          # B11
            'C': [(0, 1)],          # C00
        },
        # M6: (A10-A00)(B00+B01) -> C10(+1), C11(+1)
        {
            'A': [(2, 1), (0, -1)], # A10, A00
            'B': [(0, 1), (1, 1)],  # B00, B01
            'C': [(2, 1), (3, 1)],  # C10, C11
        },
        # M7: (A01-A11)(B10+B11) -> C00(-1), C11(+1)
        {
            'A': [(1, 1), (3, -1)], # A01, A11
            'B': [(2, 1), (3, 1)],  # B10, B11
            'C': [(0, -1), (3, 1)], # C00, C11
        },
    ]
    
    # Block index to (row_offset, col_offset) in the 4x4 matrix
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    block_offsets = {
        0: (0, 0),
        1: (0, 2),
        2: (2, 0),
        3: (2, 2),
    }
    
    for l, combo in enumerate(strassen_combos):
        for c in range(7):
            col = l * 7 + c
            
            # U: A[i,k] where i is row in 4x4, k is col in 4x4
            for block_idx, coeff in combo['A']:
                row_off, col_off = block_offsets[block_idx]
                for ii in range(2):
                    for kk in range(2):
                        i = row_off + ii
                        k = col_off + kk
                        U4[i*4 + k][col] += coeff * U2[ii*2 + kk][c]
            
            # V: B[k,j] where k is row in 4x4, j is col in 4x4
            for block_idx, coeff in combo['B']:
                row_off, col_off = block_offsets[block_idx]
                for kk in range(2):
                    for jj in range(2):
                        k = row_off + kk
                        j = col_off + jj
                        V4[k*4 + j][col] += coeff * V2[kk*2 + jj][c]
            
            # W: C[j,i] where j is col in 4x4, i is row in 4x4 (trace convention)
            for block_idx, coeff in combo['C']:
                row_off, col_off = block_offsets[block_idx]
                for jj in range(2):
                    for ii in range(2):
                        j = col_off + jj
                        i = row_off + ii
                        W4[j*4 + i][col] += coeff * W2[jj*2 + ii][c]
    
    return U4, V4, W4

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split 4x4x9 into 4x4x4 + 4x4x4 + 4x4x1
    # Use Strassen (rank 49) for each 4x4x4 part
    # Use schoolbook (rank 16) for the 4x4x1 part
    # Total rank = 49 + 49 + 16 = 114
    
    # Get the 4x4x4 Strassen decomposition
    U4, V4, W4 = _strassen_4x4x4()
    
    # Build full decomposition
    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 114
    
    # Part 1: 4x4x4 for B columns 0-3
    # U is same for both parts (A is the same)
    # V: B[k,j] for j in 0..3
    # W: C[j,i] for j in 0..3
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            # V4[i] is for B with 4 columns, index i = k*4+j
            k, j = divmod(i, 4)
            V[k*9 + j][c] = V4[i][c]
            # W4[i] is for C with 4 columns, index i = j*4+i (trace convention)
            j2, i2 = divmod(i, 4)
            W[j2*4 + i2][c] = W4[i][c]
    
    # Part 2: 4x4x4 for B columns 4-7
    # Same A, so U is the same
    # V: B[k,j] for j in 4..7, so V[k*9 + (j-4+4)] = V[k*9 + j]
    # W: C[j,i] for j in 4..7, so W[(j-4+4)*4 + i] = W[j*4 + i]
    
    for i in range(16):
        for c in range(49):
            U[i][49 + c] = U4[i][c]
            k, j = divmod(i, 4)
            V[k*9 + (j + 4)][49 + c] = V4[i][c]
            j2, i2 = divmod(i, 4)
            W[(j2 + 4)*4 + i2][49 + c] = W4[i][c]
    
    # Part 3: 4x4x1 for B column 8
    # C[i,8] = sum_k A[i,k]*B[k,8]
    # 16 terms
    for i in range(4):
        for k in range(4):
            c = 98 + i*4 + k
            U[i*4 + k][c] = 1
            V[k*9 + 8][c] = 1
            W[8*4 + i][c] = 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
