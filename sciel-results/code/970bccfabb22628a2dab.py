"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Reshape to 3D tensors
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    # Compute sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # This is a batched einsum: sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check all entries are 1
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _try_strassen_4x4_4x8():
    """
    Attempt a Strassen-based decomposition for 4x4 * 4x8.
    
    For 4x4 * 4x4, Strassen gives rank 7^2 = 49 (or 49 with the standard scheme).
    For 4x4 * 4x8, we can split the 8-column dimension into two 4-column blocks,
    giving two independent 4x4*4x4 multiplications, each of rank 49, total rank 98.
    
    However, we can potentially do better by sharing some Strassen multiplications
    across the two blocks. Let's think about this more carefully.
    
    Actually, for 4x4 * 4x8:
    - Split A (4x4) and B (4x8) into blocks
    - A is 4x4, B is 4x8 = two 4x4 blocks side by side
    - C = A * B = A * [B1 | B2] = [A*B1 | A*B2]
    - Each A*Bk is 4x4*4x4, which has Strassen rank 49
    - So total rank is 49 + 49 = 98
    
    But we can be smarter. Let's use the Strassen algorithm directly on the 4x4*4x8 problem.
    
    Strassen for 4x4*4x4:
    Split each 4x4 into 2x2 blocks. Then apply Strassen to each 2x2 multiplication.
    For 2x2*2x2, Strassen uses 7 multiplications.
    For 4x4*4x4, we have 4 block multiplications (each 2x2*2x2), each using 7 mults = 28 mults.
    But wait, that's not right either. Let me reconsider.
    
    Actually, the standard Strassen for 4x4*4x4:
    - Split into 2x2 blocks: A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    - Each block is 2x2
    - Compute 7 Strassen products for each of the 4 output blocks? No...
    
    Let me think again. For 4x4*4x4 with Strassen:
    - The 4x4 matrices are split into 2x2 blocks
    - We need to compute 4 products of 2x2 blocks
    - Each 2x2*2x2 can be done with 7 multiplications (Strassen)
    - So total = 4 * 7 = 28 multiplications? No, that's not how Strassen works.
    
    Actually, Strassen's algorithm for n x n where n is a power of 2:
    - For 2x2: 7 multiplications
    - For 4x4: split into 2x2 blocks, use 7 "recursive" multiplications of 2x2 matrices
    - Each "multiplication" of 2x2 matrices requires 7 scalar multiplications
    - So total for 4x4*4x4: 7 * 7 = 49 scalar multiplications
    
    For 4x4*4x8:
    - Split B into two 4x4 blocks: B = [B1 | B2]
    - C = A * B = [A*B1 | A*B2]
    - A*B1: 4x4*4x4, rank 49
    - A*B2: 4x4*4x4, rank 49
    - Total rank: 98
    
    Can we do better? Maybe by using a different blocking or by sharing computations.
    
    Let's try to construct the rank-98 decomposition using Strassen.
    
    Strassen for 2x2*2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4*4x4, we apply this at the block level where each "element" is a 2x2 matrix.
    
    Let me denote the 2x2 blocks of A (4x4) as A11, A12, A21, A22 (each 2x2).
    Similarly for B: B11, B12, B21, B22.
    
    The 7 Strassen products are:
    M1 = (A11+A22)(B11+B22)  -- product of two 2x2 matrices
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    Each M is a 2x2 matrix, and each requires 7 scalar multiplications (Strassen again).
    So total = 7 * 7 = 49 scalar multiplications.
    
    For the 4x4*4x8 case, we have B as 4x8, which we split into B1 (4x4) and B2 (4x4).
    We compute C1 = A*B1 and C2 = A*B2, each with rank 49, total rank 98.
    
    Let's construct this.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll build the decomposition using Strassen at two levels.
    # Level 1: Split the 4x4*4x8 into two 4x4*4x4 problems.
    # Level 2: Each 4x4*4x4 uses Strassen (7 multiplications of 2x2*2x2).
    # Level 3: Each 2x2*2x2 uses Strassen (7 scalar multiplications).
    
    # The total rank is 2 * 7 * 7 = 98.
    
    # Let's build the factor matrices.
    # U has shape (n*m, rank) = (16, 98)
    # V has shape (m*p, rank) = (32, 98)
    # W has shape (p*n, rank) = (32, 98)
    
    # Indexing:
    # A is 4x4, indexed by (i, j) where i in 0..3 (row), j in 0..3 (col)
    # B is 4x8, indexed by (j, k) where j in 0..3 (row), k in 0..7 (col)
    # C is 4x8, indexed by (i, k) where i in 0..3 (row), k in 0..7 (col)
    
    # U[i*m + j, c] = U[i*4 + j, c]
    # V[j*p + k, c] = V[j*8 + k, c]
    # W[k*n + i, c] = W[k*4 + i, c]
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # Strassen coefficients for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For a 2x2*2x2 multiplication, the Strassen decomposition has:
    # U factors (for A): each of the 7 products has a coefficient for each of the 4 elements of A
    # V factors (for B): each of the 7 products has a coefficient for each of the 4 elements of B
    # W factors (for C): each of the 4 elements of C is a linear combination of the 7 products
    
    # Let's define the Strassen coefficients for 2x2*2x2.
    # For product r (0..6), the U coefficient for A element (a, b) is:
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    
    # M0 (M1): A11+A22 -> U coeffs: A11:1, A12:0, A21:0, A22:1
    #             B11+B22 -> V coeffs: B11:1, B12:0, B21:0, B22:1
    # M1 (M2): A21+A22 -> U coeffs: A11:0, A12:0, A21:1, A22:1
    #             B11       -> V coeffs: B11:1, B12:0, B21:0, B22:0
    # M2 (M3): A11       -> U coeffs: A11:1, A12:0, A21:0, A22:0
    #             B12-B22 -> V coeffs: B11:0, B12:1, B21:0, B22:-1
    # M3 (M4): A22       -> U coeffs: A11:0, A12:0, A21:0, A22:1
    #             B21-B11 -> V coeffs: B11:-1, B12:0, B21:1, B22:0
    # M4 (M5): A11+A12 -> U coeffs: A11:1, A12:1, A21:0, A22:0
    #             B22       -> V coeffs: B11:0, B12:0, B21:0, B22:1
    # M5 (M6): A21-A11 -> U coeffs: A11:-1, A12:0, A21:1, A22:0
    #             B11+B12 -> V coeffs: B11:1, B12:1, B21:0, B22:0
    # M6 (M7): A12-A22 -> U coeffs: A11:0, A12:1, A21:0, A22:-1
    #             B21+B22 -> V coeffs: B11:0, B12:0, B21:1, B22:1
    
    strassen_U = [
        [1, 0, 0, 1],  # M0: A11+A22
        [0, 0, 1, 1],  # M1: A21+A22
        [1, 0, 0, 0],  # M2: A11
        [0, 0, 0, 1],  # M3: A22
        [1, 1, 0, 0],  # M4: A11+A12
        [-1, 0, 1, 0], # M5: A21-A11
        [0, 1, 0, -1], # M6: A12-A22
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M0: B11+B22
        [1, 0, 0, 0],  # M1: B11
        [0, 1, 0, -1], # M2: B12-B22
        [-1, 0, 1, 0], # M3: B21-B11
        [0, 0, 0, 1],  # M4: B22
        [1, 1, 0, 0],  # M5: B11+B12
        [0, 0, 1, 1],  # M6: B21+B22
    ]
    
    # W coefficients: for each output element (c, d) of the 2x2 result,
    # the coefficients of M0..M6
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
    ]
    
    # Now, for the 4x4*4x4 case, we apply Strassen at the block level.
    # The 4x4 matrices are split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2.
    
    # The 7 Strassen products at the block level are:
    # M0 = (A11+A22)(B11+B22)
    # M1 = (A21+A22)B11
    # M2 = A11(B12-B22)
    # M3 = A22(B21-B11)
    # M4 = (A11+A12)B22
    # M5 = (A21-A11)(B11+B12)
    # M6 = (A12-A22)(B21+B22)
    
    # Each M is a 2x2 matrix, computed using Strassen (7 scalar multiplications).
    
    # For the 4x4*4x8 case, we have two 4x4*4x4 subproblems.
    # Subproblem 1: A * B1 where B1 is the first 4 columns of B.
    # Subproblem 2: A * B2 where B2 is the last 4 columns of B.
    
    # Let's build the decomposition for subproblem 1 first, then subproblem 2.
    
    # For subproblem t (t=0 for B1, t=1 for B2):
    # The B matrix is 4x4, with blocks B11, B12, B21, B22.
    # For t=0: B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # For t=1: B11 = B[0:2, 4:6], B12 = B[0:2, 6:8], B21 = B[2:4, 4:6], B22 = B[2:4, 6:8]
    
    # The output C is 4x4 for each subproblem.
    # For t=0: C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    # For t=1: C11 = C[0:2, 4:6], C12 = C[0:2, 6:8], C21 = C[2:4, 4:6], C22 = C[2:4, 6:8]
    
    # Let's index the rank columns.
    # For subproblem t, Strassen product s (0..6), scalar Strassen product r (0..6):
    # Column index = t * 49 + s * 7 + r
    
    for t in range(2):  # Two subproblems
        for s in range(7):  # 7 Strassen products at the block level
            for r in range(7):  # 7 scalar Strassen products
                col = t * 49 + s * 7 + r
                
                # Determine which A elements contribute to this product.
                # The A element is determined by s (block-level Strassen) and r (scalar-level Strassen).
                # Block-level: s determines which 2x2 blocks of A are used.
                # Scalar-level: r determines which element of the 2x2 block is used.
                
                # For Strassen product s, the A blocks involved are:
                # s=0: A11, A22
                # s=1: A21, A22
                # s=2: A11
                # s=3: A22
                # s=4: A11, A12
                # s=5: A21, A11
                # s=6: A12, A22
                
                # The coefficients for each A block are given by strassen_U[s].
                # strassen_U[s] = [coeff_A11, coeff_A12, coeff_A21, coeff_A22]
                
                # For each A block with non-zero coefficient, we need to determine
                # which scalar elements of that block contribute based on r.
                
                # The scalar Strassen product r determines which element of the 2x2 block is used.
                # strassen_U[r] = [coeff for element (0,0), (0,1), (1,0), (1,1)]
                
                # A block (bi, bj) where bi is row block (0 or 1), bj is col block (0 or 1).
                # A block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1.
                # Element (ei, ej) within the block corresponds to row 2*bi+ei, col 2*bj+ej.
                
                # For scalar Strassen product r, the elements used are:
                # r=0: (0,0), (1,1) with coeffs [1, 0, 0, 1]
                # r=1: (1,0), (1,1) with coeffs [0, 0, 1, 1]
                # r=2: (0,0) with coeff [1, 0, 0, 0]
                # r=3: (1,1) with coeff [0, 0, 0, 1]
                # r=4: (0,0), (0,1) with coeffs [1, 1, 0, 0]
                # r=5: (1,0), (0,0) with coeffs [-1, 0, 1, 0]
                # r=6: (0,1), (1,1) with coeffs [0, 1, 0, -1]
                
                # For each A block (bi, bj) with block-level coefficient c_block:
                # For each element (ei, ej) within the block with scalar-level coefficient c_scalar:
                # The A element is at row 2*bi+ei, col 2*bj+ej.
                # U[2*bi+ei, 2*bj+ej, col] += c_block * c_scalar
                
                # Similarly for B and W.
                
                # B block (bi, bj) for subproblem t:
                # For t=0: B covers cols 0..3, so block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # For t=1: B covers cols 4..7, so block (bi, bj) covers rows 2*bi..2*bi+1, cols 4+2*bj..4+2*bj+1
                
                # W: The output C for subproblem t.
                # For t=0: C covers cols 0..3
                # For t=1: C covers cols 4..7
                # C block (ci, cj) covers rows 2*ci..2*ci+1, cols (4*t)+2*cj..(4*t)+2*cj+1
                
                # Let's build the U, V, W contributions.
                
                # U contributions:
                for bi in range(2):  # A row block
                    for bj in range(2):  # A col block
                        c_block = strassen_U[s][bj * 2 + bi]  # Wait, need to map correctly
                        # strassen_U[s] = [coeff_A11, coeff_A12, coeff_A21, coeff_A22]
                        # A11: bi=0, bj=0 -> index 0
                        # A12: bi=0, bj=1 -> index 1
                        # A21: bi=1, bj=0 -> index 2
                        # A22: bi=1, bj=1 -> index 3
                        # So index = bi * 2 + bj
                        idx = bi * 2 + bj
                        c_block = strassen_U[s][idx]
                        if c_block == 0:
                            continue
                        # For scalar Strassen product r, the elements used:
                        for ei in range(2):
                            for ej in range(2):
                                c_scalar = strassen_U[r][ei * 2 + ej]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei  # A row
                                j = 2 * bj + ej  # A col
                                U[i * 4 + j][col] += c_block * c_scalar
                
                # V contributions:
                for bi in range(2):  # B row block
                    for bj in range(2):  # B col block
                        idx = bi * 2 + bj
                        c_block = strassen_V[s][idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                c_scalar = strassen_V[r][ei * 2 + ej]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei  # B row
                                k = 4 * t + 2 * bj + ej  # B col (offset by 4*t for subproblem t)
                                V[j * 8 + k][col] += c_block * c_scalar
                
                # W contributions:
                # The output C element is determined by s and r.
                # For block-level Strassen product s, the C blocks involved are:
                # s=0: C11, C22 (coeffs: C11:1, C22:1) -- wait, let me recheck
                # Actually, the W coefficients are for the scalar-level output.
                # The block-level output is a 2x2 matrix of 2x2 blocks.
                # For Strassen product s, the contribution to each C block is:
                # We need to figure out which C blocks are affected by product s.
                
                # Let me reconsider. The Strassen algorithm computes:
                # C11 = M0 + M3 - M4 + M6
                # C12 = M2 + M4
                # C21 = M1 + M3
                # C22 = M0 - M1 + M2 + M5
                
                # So for product s, the C blocks affected are:
                # s=0 (M0): C11 (coeff 1), C22 (coeff 1)
                # s=1 (M1): C21 (coeff 1), C22 (coeff -1)
                # s=2 (M2): C12 (coeff 1), C22 (coeff 1)
                # s=3 (M3): C11 (coeff 1), C21 (coeff 1)
                # s=4 (M4): C11 (coeff -1), C12 (coeff 1)
                # s=5 (M5): C22 (coeff 1)
                # s=6 (M6): C11 (coeff 1)
                
                # So for each C block (ci, cj) affected by product s with coefficient c_cblock:
                # For scalar Strassen product r, the elements of the C block are:
                # strassen_W[r] = [coeff for C element (0,0), (0,1), (1,0), (1,1)]
                
                # C block (ci, cj) covers rows 2*ci..2*ci+1, cols (4*t)+2*cj..(4*t)+2*cj+1
                
                # Let's define the C block coefficients for each s.
                c_block_coeffs = [
                    [(0, 0, 1), (1, 1, 1)],  # s=0: C11 (1), C22 (1)
                    [(1, 0, 1), (1, 1, -1)], # s=1: C21 (1), C22 (-1)
                    [(0, 1, 1), (1, 1, 1)],  # s=2: C12 (1), C22 (1)
                    [(0, 0, 1), (1, 0, 1)],  # s=3: C11 (1), C21 (1)
                    [(0, 0, -1), (0, 1, 1)], # s=4: C11 (-1), C12 (1)
                    [(1, 1, 1)],              # s=5: C22 (1)
                    [(0, 0, 1)],              # s=6: C11 (1)
                ]
                
                for ci, cj, c_cblock in c_block_coeffs[s]:
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = strassen_W[r][ei * 2 + ej]
                            if c_scalar == 0:
                                continue
                            i = 2 * ci + ei  # C row
                            k = 4 * t + 2 * cj + ej  # C col
                            W[k * 4 + i][col] += c_cblock * c_scalar
    
    return {"U": U, "V": V, "W": W}

def construct():
    # Try Strassen-based decomposition first
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
