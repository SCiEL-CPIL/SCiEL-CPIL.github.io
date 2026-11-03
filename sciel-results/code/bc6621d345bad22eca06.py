"""Improved program: hybrid 4x4 Strassen + 2x2 Strassen for 4x4 * 4x6."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    Convention: a = i*2+j, b = j*2+k, c = k*2+i for 2x2 matrices."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)*(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)*B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00*(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11*(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)*B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)*(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)*(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
    return U, V, W

def _strassen_4x4():
    """Return 49-term recursive Strassen decomposition for 4x4 matrix multiplication.
    Convention: a = i*4+j, b = j*4+k, c = k*4+i for 4x4 matrices.
    Built by applying Strassen at block level (2x2 blocks of 2x2),
    each block product uses 2x2 Strassen (rank 7), giving 7*7=49 terms."""
    U1, V1, W1 = _strassen_2x2()
    
    # Strassen block formulas for 4x4 (splitting into 2x2 blocks of 2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product of 2x2 blocks, decomposed via Strassen (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # For each of the 7 Strassen block-products, we have 7 sub-terms.
    # The U matrix has 16 rows (4x4 A elements), V has 16 rows, W has 16 rows.
    
    # Block index mapping for 4x4:
    # A block (bi,bj) has elements at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A[row,col] -> block (row//2, col//2), local (row%2, col%2)
    # a_idx = row*4 + col = (2*bi+li)*4 + (2*bj+lj)
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen block product definitions:
    # For each block product M_s (s=0..6), define:
    #   A-blocks involved and their coefficients
    #   B-blocks involved and their coefficients
    #   C-blocks that receive contributions and their coefficients
    
    # M1 = (A00+A11)*(B00+B11): A blocks [(0,0,+1),(1,1,+1)], B blocks [(0,0,+1),(1,1,+1)]
    #   C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00: A blocks [(1,0,+1),(1,1,+1)], B blocks [(0,0,+1)]
    #   C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11): A blocks [(0,0,+1)], B blocks [(0,1,+1),(1,1,-1)]
    #   C01 += M3, C11 += M3
    # M4 = A11*(B10-B00): A blocks [(1,1,+1)], B blocks [(1,0,+1),(0,0,-1)]
    #   C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11: A blocks [(0,0,+1),(0,1,+1)], B blocks [(1,1,+1)]
    #   C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01): A blocks [(1,0,+1),(0,0,-1)], B blocks [(0,0,+1),(0,1,+1)]
    #   C11 += M6
    # M7 = (A01-A11)*(B10+B11): A blocks [(0,1,+1),(1,1,-1)], B blocks [(1,0,+1),(1,1,+1)]
    #   C00 += M7, C01 += M7
    
    strassen_defs = [
        # (A_blocks, B_blocks, C_blocks)
        # Each: list of (block_row, block_col, coeff)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),   # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),             # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)]),     # M7
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        base_col = s * 7
        for sub_r in range(7):
            col = base_col + sub_r
            # U: for each A block involved, add U1[local_idx, sub_r] * coeff
            for (br, bc, coeff) in a_blocks:
                for li in range(2):
                    for lj in range(2):
                        row = 2*br + li
                        col_a = 2*bc + lj
                        a_idx = row * 4 + col_a
                        local_idx = li * 2 + lj
                        U[a_idx, col] += U1[local_idx, sub_r] * coeff
            # V: for each B block involved
            for (br, bc, coeff) in b_blocks:
                for li in range(2):
                    for lj in range(2):
                        row = 2*br + li
                        col_b = 2*bc + lj
                        b_idx = row * 4 + col_b
                        local_idx = li * 2 + lj
                        V[b_idx, col] += V1[local_idx, sub_r] * coeff
            # W: for each C block involved
            for (br, bc, coeff) in c_blocks:
                for li in range(2):
                    for lj in range(2):
                        row = 2*br + li
                        col_c = 2*bc + lj
                        c_idx = row * 4 + col_c
                        local_idx = li * 2 + lj
                        W[c_idx, col] += W1[local_idx, sub_r] * coeff
    
    return U, V, W

def _hybrid_4x4_4x6():
    """Hybrid: 4x4 Strassen for cols 0-3, 2x2 Strassen blocks for cols 4-5.
    C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4, rank 49)
    C[:, 4:6] = A * B[:, 4:6]  (4x4 * 4x2, rank 2*14=28)
    Total rank: 77."""
    n, m, p = 4, 4, 6
    U4, V4, W4 = _strassen_4x4()  # 16x49, 16x49, 16x49
    U2, V2, W2 = _strassen_2x2()  # 4x7, 4x7, 4x7
    
    total_rank = 49 + 28
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    # Part 1: 4x4 * 4x4 for columns 0-3
    # A: 16 elements, index a = row*4+col
    # B[:, 0:4]: 16 elements, index b = row*4+col (within the 4x4 submatrix)
    #   Global B index: row*6 + col
    # C[:, 0:4]: 16 elements, index c = row*4+col (within the 4x4 submatrix)
    #   Global C index: row*6 + col
    
    U[:, 0:49] = U4
    # V4 has 16 rows for B[:,0:4], map to global B indices
    for r in range(16):
        b_row = r // 4
        b_col = r % 4
        V[b_row * 6 + b_col, 0:49] = V4[r, :]
    # W4 has 16 rows for C[:,0:4], map to global C indices
    for r in range(16):
        c_row = r // 4
        c_col = r % 4
        W[c_row * 6 + c_col, 0:49] = W4[r, :]
    
    # Part 2: 4x4 * 4x2 for columns 4-5
    # Split into two 2-column strips: cols 4-5
    # For each column pair (cj in {4,5}), do 4x4 * 4x1... 
    # Actually, treat as two separate 2x2 column blocks:
    # B[:, 4:6] is 4x2. Split into two 2x1 column blocks? No, 2x2 doesn't fit.
    # Better: split B[:, 4:6] into two 2x1 blocks won't work with 2x2 Strassen.
    # Instead: split into top 2 rows and bottom 2 rows of B[:, 4:6], each 2x2.
    # B[:, 4:6] = [B_top; B_bottom] where B_top is 2x2 (rows 0-1, cols 4-5), B_bottom is 2x2 (rows 2-3, cols 4-5)
    # C[:, 4:6] = A[:, 0:2] * B_top + A[:, 2:4] * B_bottom
    # A[:, 0:2] is 4x2, A[:, 2:4] is 4x2.
    # This is sum of two 4x2 * 2x2 products... not directly 2x2 Strassen.
    
    # Alternative: treat as 4x4 * 4x2 by padding B to 4x4 with zeros? No, that changes rank.
    # Better approach: for the 4x4 * 4x2 part, use two independent 2x2 Strassen decompositions:
    # Split A into 2x2 blocks: A00, A01 (top row of blocks), A10, A11 (bottom row)
    # B[:, 4:6] into 2x2 blocks: B0 (rows 0-1, cols 4-5), B1 (rows 2-3, cols 4-5)
    # C[:, 4:6] into 2x2 blocks: C0 (rows 0-1, cols 4-5), C1 (rows 2-3, cols 4-5)
    # C0 = A00*B0 + A01*B1  (2x2 * 2x2 + 2x2 * 2x2, rank 14)
    # C1 = A10*B0 + A11*B1  (2x2 * 2x2 + 2x2 * 2x2, rank 14)
    # Total for this part: 28
    
    for ci in range(2):  # C block row
        base_col = 49 + ci * 14
        for local_r in range(14):
            col = base_col + local_r
            if local_r < 7:
                # First 7 terms: A_top_block * B0
                # A block (ci, 0): rows 2*ci..2*ci+1, cols 0..1
                # B0: rows 0..1, cols 4..5
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + lj
                        U[a_idx, col] += U2[li*2+lj, local_r]
                for li in range(2):
                    for lj in range(2):
                    # B0 local (li, lj) -> global row li, col 4+lj
                        b_idx = li * 6 + (4 + lj)
                        V[b_idx, col] += V2[li*2+lj, local_r]
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (4 + lj)
                        W[c_idx, col] += W2[li*2+lj, local_r]
            else:
                # Next 7 terms: A_top_block * B1
                # A block (ci, 1): rows 2*ci..2*ci+1, cols 2..3
                # B1: rows 2..3, cols 4..5
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + (2 + lj)
                        U[a_idx, col] += U2[li*2+lj, local_r - 7]
                for li in range(2):
                    for lj in range(2):
                        b_idx = (2 + li) * 6 + (4 + lj)
                        V[b_idx, col] += V2[li*2+lj, local_r - 7]
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (4 + lj)
                        W[c_idx, col] += W2[li*2+lj, local_r - 7]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Try hybrid with rank 77
    result = _hybrid_4x4_4x6()
    
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
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
