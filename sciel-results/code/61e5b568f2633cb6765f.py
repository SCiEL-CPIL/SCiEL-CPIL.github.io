"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n+i, r] corresponds to output element C[i,k] from A[i,j]*B[j,k]
    The decomposition requires: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    # For each (i,j,k) triple, check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Classic Strassen 2x2x2 decomposition, rank 7.
    Returns U (4x7), V (4x7), W (4x7) for multiplying 2x2 * 2x2.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[i*2+j, r] = coefficient for A[i,j] in multiplication r
    # V[j*2+k, r] = coefficient for B[j,k] in multiplication r
    # W[k*2+i, r] = coefficient for C[i,k] from multiplication r
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1   # A00, A11
    V[0][0] = 1; V[3][0] = 1   # B00, B11
    W[0][0] = 1; W[3][0] = 1   # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1   # A10, A11
    V[0][1] = 1                # B00
    W[2][1] = 1; W[3][1] = -1  # C10, C11
    
    # M3 = A00(B10-B11)
    U[0][2] = 1                # A00
    V[2][2] = 1; V[3][2] = -1  # B10, B11
    W[1][2] = 1; W[3][2] = 1   # C01, C11
    
    # M4 = A11(B01-B10)
    U[3][3] = 1                # A11
    V[1][3] = 1; V[2][3] = -1  # B01, B10
    W[0][3] = 1; W[2][3] = 1   # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1   # A00, A01
    V[3][4] = 1                # B11
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, A00
    V[0][5] = 1; V[1][5] = 1   # B00, B01
    W[3][5] = 1                # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, A11
    V[2][6] = 1; V[3][6] = 1   # B10, B11
    W[0][6] = 1                # C00
    
    return U, V, W

def _pad_and_decompose(n, m, p, N, M, P, U2, V2, W2):
    """
    Decompose n x m * m x p by padding to N x M * M x P and using the 2x2x2 Strassen
    recursively, then extracting the relevant sub-tensor.
    """
    # For now, we'll try a direct approach: use Strassen on blocks
    # This is complex, so let's try a different strategy.
    pass

def _strassen_pad(n, m, p):
    """
    Try to use Strassen by padding n,m,p to powers of 2.
    4x5*5x6 -> pad to 4x8*8x8 (m=5->8, p=6->8)
    Then 4x8*8x8 can be decomposed using Strassen on 2x2 blocks.
    4 = 2*2, 8 = 2*2*2, so we need 3 levels of Strassen in the 8-dimension.
    Actually, let's think more carefully.
    
    We want to multiply A (4x5) by B (5x6) to get C (4x6).
    Pad A to 4x8, B to 8x8, compute C' = A' * B' (4x8), extract C (4x6).
    
    4x8 * 8x8: We can view this as 2x(2*4) * (2*4)*8... this gets complicated.
    
    Alternative: pad to 8x8 * 8x8 and use 3 levels of Strassen (rank 7^3 = 343).
    That's worse than 120.
    
    Better: 4x8 * 8x8. View as (2x2) blocks:
    A is 2x4 blocks of 2x2, B is 4x4 blocks of 2x2.
    C is 2x4 blocks of 2x2.
    C[i,j] = sum_k A[i,k] * B[k,j] for block indices.
    This is a 2x4 * 4x4 block matmul, which has 2*4*4 = 32 block multiplications,
    each of rank 7, giving total rank 224. Worse.
    
    Let me try a different approach: direct construction for small cases.
    """
    pass

def _try_bilinear_search(n, m, p, max_rank, time_limit):
    """
    Try to find a decomposition of rank <= max_rank using a greedy/greedy-random approach.
    This is a heuristic search.
    """
    start = time.time()
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # For small ranks, we can try to verify candidate decompositions
    # But constructing them is hard. Let's try a different approach.
    
    # Use the fact that we can combine Strassen with schoolbook for sub-blocks.
    # For 4x5*5x6:
    # Split A into 2x2 blocks: A = [[A00, A01, A02, A03, A04], [A10, A11, A12, A13, A14]]
    # where each block is 2x1 (since 4 rows = 2 blocks of 2, 5 cols = 5 blocks of 1)
    # This doesn't work well.
    
    # Alternative: use the "laser method" or known results.
    # For 4x5*5x6, the best known upper bound is likely around 100-110.
    # Let's try a structured decomposition.
    
    # Strategy: pad 4x5*5x6 to 4x8*8x8, use a 2-level Strassen-like approach.
    # 4x8*8x8: split into 2x4 blocks of 2x2.
    # A has 2 row-blocks, 4 col-blocks (each 2x2)
    # B has 4 row-blocks, 4 col-blocks (each 2x2)
    # C has 2 row-blocks, 4 col-blocks (each 2x2)
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # This is a 2x4 * 4x4 matrix multiplication of 2x2 blocks.
    # Rank of 2x4*4x4 block matmul: schoolbook is 2*4*4 = 32, each rank 7 => 224.
    # Too high.
    
    # Let me try: 4x5*5x6 directly with a smarter decomposition.
    # Known: M(4,5,6) <= M(4,8,8) and M(4,8,8) can be computed via:
    # Pad to 8x8*8x8, use 3-level Strassen: rank 343. Too high.
    
    # Actually, let's just return the schoolbook for now and try to improve.
    # The key insight: we can use Strassen on the 5x6 part by padding 5->8, 6->8.
    # 4x8 * 8x8: this is a rectangular matmul.
    # We can compute it as: view A as 2x2 blocks (2 row-blocks, 4 col-blocks of 2x2)
    # B as 4x4 blocks of 2x2.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # For fixed i (2 choices), j (4 choices): C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    # This is 2*4 = 8 independent "vector" multiplications of length 4.
    # Each can be done with rank... well, it's a sum of 4 products.
    # Schoolbook: 8*4 = 32 multiplications of 2x2, each rank 7 => 224.
    
    # I think for this problem size, the best we can do with known methods is still
    # close to schoolbook. Let me just return schoolbook.
    
    return best, best_rank

def _strassen_2x2x2_block():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) with indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _outer_product_2x1_1x2():
    """Rank-4 decomposition for (2x1)*(1x2) -> 2x2 outer product.
    U (2x4), V (2x4), W (4x4) with indexing U[i*1+j,r]=U[i,r], V[j*1+k,r]=V[k,r], W[k*2+i,r]
    Actually for 2x1 * 1x2: A is 2x1, B is 1x2, C is 2x2.
    C[i,k] = A[i,0]*B[0,k]
    This is an outer product, rank 4 (schoolbook).
    U[i*1+0, r] = U[i, r], V[0*1+k, r] = V[k, r], W[k*2+i, r]
    """
    U = [[0]*4 for _ in range(2)]  # 2x1 matrix, indexed by i
    V = [[0]*4 for _ in range(2)]  # 1x2 matrix, indexed by k
    W = [[0]*4 for _ in range(4)]  # 2x2 matrix, indexed by k*2+i
    
    # C[i,k] = A[i]*B[k]
    # For each (i,k), we need U[i]*V[k]*W[k*2+i] = 1
    # Use 4 rank-1 terms:
    for i in range(2):
        for k in range(2):
            col = i*2 + k
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
    
    return U, V, W

def _block_decompose_4x5x6():
    """
    Decompose 4x5 * 5x6 using block structure with Strassen on 2x2 blocks.
    Split: rows 4=2+2, cols 5=2+2+1, cols 6=2+2+2
    
    Block structure:
    A: 2 row-blocks x 3 col-blocks
      A[0,0]: 2x2, A[0,1]: 2x2, A[0,2]: 2x1
      A[1,0]: 2x2, A[1,1]: 2x2, A[1,2]: 2x1
    
    B: 3 row-blocks x 3 col-blocks
      B[0,0]: 2x2, B[0,1]: 2x2, B[0,2]: 2x2
      B[1,0]: 2x2, B[1,1]: 2x2, B[1,2]: 2x2
      B[2,0]: 1x2, B[2,1]: 1x2, B[2,2]: 1x2
    
    C: 2 row-blocks x 3 col-blocks
      C[0,0]: 2x2, C[0,1]: 2x2, C[0,2]: 2x2
      C[1,0]: 2x2, C[1,1]: 2x2, C[1,2]: 2x2
    
    C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j]
    
    For (i,k) with A[i,k] being 2x2 and B[k,j] being 2x2: use Strassen (rank 7)
    For (i,k) with A[i,k] being 2x1 and B[k,j] being 1x2: use outer product (rank 4)
    """
    # Get Strassen components
    U_s, V_s, W_s = _strassen_2x2x2_block()
    U_o, V_o, W_o = _outer_product_2x1_1x2()
    
    # Define block sizes
    # A blocks: row_splits = [2,2], col_splits = [2,2,1]
    # B blocks: row_splits = [2,2,1], col_splits = [2,2,2]
    # C blocks: row_splits = [2,2], col_splits = [2,2,2]
    
    a_row_splits = [2, 2]
    a_col_splits = [2, 2, 1]
    b_row_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    c_row_splits = [2, 2]
    c_col_splits = [2, 2, 2]
    
    # Precompute offsets
    a_row_off = [0]
    for s in a_row_splits[:-1]:
        a_row_off.append(a_row_off[-1] + s)
    a_col_off = [0]
    for s in a_col_splits[:-1]:
        a_col_off.append(a_col_off[-1] + s)
    b_row_off = [0]
    for s in b_row_splits[:-1]:
        b_row_off.append(b_row_off[-1] + s)
    b_col_off = [0]
    for s in b_col_splits[:-1]:
        b_col_off.append(b_col_off[-1] + s)
    c_row_off = [0]
    for s in c_row_splits[:-1]:
        c_row_off.append(c_row_off[-1] + s)
    c_col_off = [0]
    for s in c_col_splits[:-1]:
        c_col_off.append(c_col_off[-1] + s)
    
    # Count total rank
    total_rank = 0
    for i in range(2):
        for k in range(3):
            a_height = a_row_splits[i]
            a_width = a_col_splits[k]
            b_height = b_row_splits[k]
            for j in range(3):
                b_width = b_col_splits[j]
                if a_width == 2 and b_height == 2 and b_width == 2:
                    total_rank += 7
                elif a_width == 1 and b_height == 1 and b_width == 2:
                    total_rank += 4
                else:
                    # Should not happen with our splits
                    total_rank += a_height * a_width * b_width
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(4 * 5)]  # n*m = 20
    V = [[0] * total_rank for _ in range(5 * 6)]  # m*p = 30
    W = [[0] * total_rank for _ in range(6 * 4)]  # p*n = 24
    
    col = 0
    for i in range(2):
        for k in range(3):
            a_height = a_row_splits[i]
            a_width = a_col_splits[k]
            b_height = b_row_splits[k]
            for j in range(3):
                b_width = b_col_splits[j]
                
                a_i_start = a_row_off[i]
                a_k_start = a_col_off[k]
                b_k_start = b_row_off[k]
                b_j_start = b_col_off[j]
                c_i_start = c_row_off[i]
                c_j_start = c_col_off[j]
                
                if a_width == 2 and b_height == 2 and b_width == 2:
                    # Use Strassen
                    for r in range(7):
                        # U: A[i,k] block, index (a_i_start + ii, a_k_start + jj) -> row a_i_start+ii, col a_k_start+jj
                        # U row index: (a_i_start+ii)*5 + (a_k_start+jj)
                        # V row index: (b_k_start+kk)*6 + (b_j_start+ll)
                        # W row index: (c_j_start+ll)*4 + (c_i_start+mm)
                        for ii in range(2):
                            for jj in range(2):
                                if U_s[ii*2+jj][r] != 0:
                                    u_row = (a_i_start + ii) * 5 + (a_k_start + jj)
                                    U[u_row][col] += U_s[ii*2+jj][r]
                        for kk in range(2):
                            for ll in range(2):
                                if V_s[kk*2+ll][r] != 0:
                                    v_row = (b_k_start + kk) * 6 + (b_j_start + ll)
                                    V[v_row][col] += V_s[kk*2+ll][r]
                        for ll in range(2):
                            for mm in range(2):
                                if W_s[ll*2+mm][r] != 0:
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + mm)
                                    W[w_row][col] += W_s[ll*2+mm][r]
                        col += 1
                elif a_width == 1 and b_height == 1 and b_width == 2:
                    # Use outer product
                    for r in range(4):
                        for ii in range(2):
                            if U_o[ii][r] != 0:
                                u_row = (a_i_start + ii) * 5 + a_k_start
                                U[u_row][col] += U_o[ii][r]
                        for ll in range(2):
                            if V_o[ll][r] != 0:
                                v_row = b_k_start * 6 + (b_j_start + ll)
                                V[v_row][col] += V_o[ll][r]
                        for ll in range(2):
                            for mm in range(2):
                                if W_o[ll*2+mm][r] != 0:
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + mm)
                                    W[w_row][col] += W_o[ll*2+mm][r]
                        col += 1
                else:
                    # Fallback: schoolbook for this block
                    for ii in range(a_height):
                        for jj in range(a_width):
                            for kk in range(b_height):
                                for ll in range(b_width):
                                    u_row = (a_i_start + ii) * 5 + (a_k_start + jj)
                                    v_row = (b_k_start + kk) * 6 + (b_j_start + ll)
                                    w_row = (c_j_start + ll) * 4 + (c_i_start + ii)
                                    U[u_row][col] = 1
                                    V[v_row][col] = 1
                                    W[w_row][col] = 1
                                    col += 1
    
    return {"U": U, "V": V, "W": W}, total_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # safety margin
    
    # Try block decomposition with Strassen
    try:
        result, rank = _block_decompose_4x5x6()
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
