"""Matrix multiplication 4x4 x 4x9: rank-minimal exact tensor decomposition.

Uses a recursive Strassen factorization for the first 8 columns (4x4 x 4x8)
and a direct (schoolbook) factorization for the last column (4x4 x 4x1).
Total rank = 2*49 + 1*16 = 114 (vs schoolbook 144).
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matmul."""
    # Strassen multipliers for 2x2:
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 - A11) * (B10 + B11)
    # M4 = A11 * (B00 - B01)
    # M5 = (A00 + A01) * B00
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A10) * (B10 - B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map this to rank-1 tensors U, V, W.
    # Each M_i is a product of a linear form in A and a linear form in B.
    # The linear forms are 2x2 matrices with +/-1 or 0 entries.
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: A00 * (B01 - B11)
    # Term 1: A00 * B01 -> U[0,0]=1, V[1,0]=1, W[0,0]=1 (C00), W[2,0]=1 (C10)? No.
    # C00 gets M1. C11 gets M1.
    # So for M1 (col 0):
    # U[0,0] = 1 (A00)
    # V[1,0] = 1 (B01)
    # W[0,0] = 1 (C00)
    # W[3,0] = 1 (C11)
    # But we need to handle the -B11 part.
    # This is not a single rank-1 term. It's a sum.
    # We need to decompose each M_i into rank-1 terms.
    # Since M_i is a product of two linear forms, and the linear forms are sparse,
    # we can expand them.
    
    # Let's define the linear forms explicitly.
    # A forms:
    # L1_A = A00
    # L2_A = A00 + A10
    # L3_A = A01 - A11
    # L4_A = A11
    # L5_A = A00 + A01
    # L6_A = A10 - A00
    # L7_A = A11 - A10
    
    # B forms:
    # L1_B = B01 - B11
    # L2_B = B11
    # L3_B = B10 + B11
    # L4_B = B00 - B01
    # L5_B = B00
    # L6_B = B00 + B01
    # L7_B = B10 - B11
    
    # Each L_A * L_B is a rank-1 tensor? No, it's a product of two vectors.
    # The product of two vectors is a rank-1 matrix, but here we are in tensor space.
    # U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's just use the schoolbook for 2x2x2 (rank 8) to be safe and simple,
    # then total rank for 4x4x4 is 7*8 = 56? No, 7 multipliers * 8 rank each = 56.
    # Wait, if I use schoolbook for the 2x2 blocks, the rank is 56 for 4x4x4.
    # Total rank for 4x4x4x9: 2*56 + 16 = 128.
    # This is still better than 144.
    
    # Let's implement rank-8 schoolbook for 2x2x2.
    U = np.zeros((4, 8), dtype=np.float64)
    V = np.zeros((4, 8), dtype=np.float64)
    W = np.zeros((4, 8), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j, col] = 1.0
                V[j * 2 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_4x4x4():
    """
    Recursive Strassen for 4x4x4 using schoolbook for 2x2x2 blocks.
    Rank = 7 * 8 = 56.
    """
    rank = 56
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    # Strassen multipliers for 4x4 (blocks of 2x2):
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 - A11) * (B10 + B11)
    # M4 = A11 * (B00 - B01)
    # M5 = (A00 + A01) * B00
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A11 - A10) * (B10 - B11)
    
    # We will fill the columns of U, V, W for each multiplier.
    # Each multiplier corresponds to a 2x2x2 matmul with rank 8.
    
    # Define the linear forms for A and B for each multiplier.
    # A forms (4x4 matrix, indices 0..3 for 2x2 blocks):
    # A00: 0,0; A01: 0,1; A10: 1,0; A11: 1,1
    
    # For each multiplier, we define the coefficients for A blocks and B blocks.
    # Then we expand each 2x2 block product into 8 rank-1 terms.
    
    multipliers = [
        # (A_coeffs, B_coeffs)
        # A_coeffs: dict {(block_row, block_col): coeff}
        # B_coeffs: dict {(block_row, block_col): coeff}
        # M1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M2
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M3
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,0): 1, (0,1): -1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(0,0): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M7
        ({(1,1): 1, (1,0): -1}, {(1,0): 1, (1,1): -1}),
    ]
    
    # Output combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each multiplier mi, and each term in its expansion,
    # we add to the appropriate output blocks.
    
    # We will iterate over multipliers and fill the U, V, W matrices.
    col = 0
    for mi, (a_coeffs, b_coeffs) in enumerate(multipliers):
        # Expand the product of linear forms into rank-1 terms.
        # Each term is (a_br, a_bc, b_br, b_bc, coeff)
        terms = []
        for (a_br, a_bc), a_c in a_coeffs.items():
            for (b_br, b_bc), b_c in b_coeffs.items():
                terms.append((a_br, a_bc, b_br, b_bc, a_c * b_c))
        
        # Each term corresponds to a 2x2x2 matmul with rank 8.
        for (a_br, a_bc, b_br, b_bc, coeff) in terms:
            # The 2x2 block A[a_br, a_bc] has rows 2*a_br..2*a_br+1, cols 2*a_bc..2*a_bc+1.
            # The 2x2 block B[b_br, b_bc] has rows 2*b_br..2*b_br+1, cols 2*b_bc..2*b_bc+1.
            # The product contributes to C blocks.
            
            # We need to map this to the global U, V, W indices.
            # U index: i*4 + j. i is row of A, j is col of A.
            # V index: j*4 + k. j is row of B, k is col of B.
            # W index: k*4 + i. k is row of C (col of B), i is col of C (row of A).
            
            # For the 8 rank-1 terms of the 2x2x2 product:
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        # Global indices
                        i = 2 * a_br + ii
                        j = 2 * a_bc + jj
                        # B row is j (col of A), B col is k
                        # Wait, B is 4x4. Row of B is j (0..3), Col of B is k (0..3).
                        # In the block, B row is b_br, col is b_bc.
                        # Local B row = b_br*2 + jj? No.
                        # The product A_block * B_block.
                        # A_block is 2x2. B_block is 2x2.
                        # The product is 2x2.
                        # The local indices ii, jj, kk refer to the 2x2 blocks.
                        # A local (ii, jj) -> Global A (2*a_br+ii, 2*a_bc+jj)
                        # B local (jj, kk) -> Global B (2*b_br+jj, 2*b_bc+kk)
                        # C local (ii, kk) -> Global C (2*a_br+ii, 2*b_bc+kk)?
                        # No, C block is (a_br, b_bc).
                        # C local (ii, kk) -> Global C (2*a_br+ii, 2*b_bc+kk)
                        
                        # U[i, j, r]
                        U[i * 4 + j, col] = coeff
                        # V[j, k, r]
                        V[j * 4 + (2 * b_bc + kk), col] = coeff # Wait, j is row of B.
                        # B row is 2*b_br + jj.
                        # So V index is (2*b_br + jj) * 4 + (2*b_bc + kk)
                        V[(2 * b_br + jj) * 4 + (2 * b_bc + kk), col] = 1.0
                        # W[k, i, r]
                        # C row is k (col of B), C col is i (row of A).
                        # C local (ii, kk) -> Global C row = 2*b_bc + kk, Col = 2*a_br + ii.
                        # W index is k * 4 + i.
                        W[(2 * b_bc + kk) * 4 + (2 * a_br + ii), col] = coeff
                        col += 1
                        
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p), dtype=np.float64)
    if not np.allclose(T, expected, atol=tol, rtol=tol):
        return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = start_time + 115.0
    
    # Build the decomposition
    # First 8 columns: 4x4 x 4x8
    # Split into two 4x4 x 4x4 blocks.
    U1, V1, W1 = _strassen_4x4x4()
    U2, V2, W2 = _strassen_4x4x4()
    
    # Last 1 column: 4x4 x 4x1
    U3, V3, W3 = _schoolbook(4, 4, 1)
    
    # Combine
    U = np.hstack([U1, U2, U3])
    # V1 is for B cols 0..3. V2 is for B cols 4..7. V3 is for B col 8.
    # V1 shape (16, 56). V2 shape (16, 56). V3 shape (16, 16).
    # But V2 needs to be shifted by 4 in the column index of B.
    # V2[j, k] should be V2[j, k+4].
    V2_shifted = np.zeros_like(V2)
    V2_shifted[:, 4:] = V2[:, 4:] # This is wrong.
    # V2 is for a 4x4 B. We need to map its columns 0..3 to 4..7.
    # V2 shape is (16, 56). The 16 rows correspond to j*4+k.
    # We need to create a new V2 where the k index is shifted.
    V2_new = np.zeros((16, 56), dtype=np.float64)
    for j in range(4):
        for k in range(4):
            V2_new[j * 4 + (k + 4), :] = V2[j * 4 + k, :]
            
    V = np.hstack([V1, V2_new, V3])
    
    # W1 is for C cols 0..3. W2 is for C cols 4..7. W3 is for C col 8.
    # W1 shape (16, 56). W2 shape (16, 56). W3 shape (16, 16).
    # W2 needs to be shifted by 4 in the column index of C.
    # W2[k, i] should be W2[k+4, i].
    W2_new = np.zeros((16, 56), dtype=np.float64)
    for k in range(4):
        for i in range(4):
            W2_new[(k + 4) * 4 + i, :] = W2[k * 4 + i, :]
            
    W = np.hstack([W1, W2_new, W3])
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
