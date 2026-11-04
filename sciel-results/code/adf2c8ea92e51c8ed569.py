import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Strategy: Use 2x2 block structure on 6x6 matrices (3x3 grid of 2x2 blocks)
    # Apply Strassen's algorithm (rank 7) to each 2x2 block multiplication
    # Total rank = 9 blocks * 7 Strassen terms = 63
    # Wait, that's not right. Let me reconsider.
    #
    # For 6x6 = 3x3 blocks of 2x2:
    # We have a 3x3 grid of 2x2 block multiplications.
    # Each 2x2 block product C[i,j] = sum_k A[i,k] * B[k,j]
    # where A[i,k], B[k,j], C[i,j] are 2x2 matrices.
    #
    # If we apply Strassen to each of the 9 block positions, we need 9*7 = 63 terms.
    # But Strassen requires the block matrices to be combined in specific ways.
    #
    # Actually, the standard approach is:
    # - Partition 6x6 into 3x3 blocks of 2x2
    # - The matrix multiplication is a 3x3 "block matrix" multiplication
    # - We can apply a fast algorithm to the 3x3 block structure
    #
    # Option 1: Schoolbook on 3x3 blocks: 27 block multiplications, each using Strassen (7 terms)
    #   Total: 27 * 7 = 189 terms (this is what the current program does)
    #
    # Option 2: Apply Laderman (rank 19) to the 3x3 block structure, where each "entry" 
    #   is a 2x2 matrix product done via Strassen (7 terms)
    #   Total: 19 * 7 = 133 terms
    #
    # Option 3: Apply Strassen recursively. 6x6 can be padded to 8x8, then:
    #   8x8 = 2x2 blocks of 4x4
    #   4x4 = 2x2 blocks of 2x2
    #   Using Strassen at each level: 7 * 7 = 49 for 8x8
    #   But we only need 6x6, so we waste some computation.
    #   The rank would still be 49 but we'd have extra zero rows/cols.
    #   Actually, the tensor rank for 6x6 padded to 8x8 would be 49, but we need to
    #   handle the padding carefully.
    #
    # Let me go with Option 2: Laderman (rank 19) for 3x3 block structure + Strassen (rank 7) for 2x2
    # Total rank = 19 * 7 = 133
    #
    # I need a correct Laderman decomposition. Let me use a verified one.
    #
    # Laderman's 19-term algorithm for 3x3 matrix multiplication:
    # Reference: Laderman, "A 19-fold bilinear algorithm for multiplying 3x3 matrices", 1976
    #
    # I'll define it carefully and verify numerically.
    
    # Define Laderman's 19 terms.
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping flat index (i*3+j) to coefficient
    # B_coeffs: dict mapping flat index (i*3+j) to coefficient  
    # C_coeffs: dict mapping flat index (i*3+j) to coefficient
    #
    # The 19 products (using 1-indexed from paper, converting to 0-indexed):
    # P1  = (a11+a21)(b11+b21)
    # P2  = (a11-a21)(b11-b21)
    # P3  = (a22+a32)(b21+b31)
    # P4  = (a22-a32)(b21-b31)
    # P5  = (a12+a13)(b32+b33)
    # P6  = (a12-a13)(b32-b33)
    # P7  = (a31+a33)(b12+b13)
    # P8  = (a31-a33)(b12-b13)
    # P9  = (a11+a22+a33)(b11+b22+b33)
    # P10 = (a21+a33)(b31+b12)
    # P11 = (a31+a22)(b21+b13)
    # P12 = (a12+a33)(b32+b11)
    # P13 = (a13+a22)(b23+b11)
    # P14 = (a23+a12)(b12+b22)
    # P15 = (a32+a13)(b13+b22)
    # P16 = (a33+a21)(b21+b13)
    # P17 = (a11+a33)(b11+b33)
    # P18 = (a21+a33)(b31+b12)  -- this seems like a duplicate of P10
    # P19 = (a31+a22)(b21+b13)  -- this seems like a duplicate of P11
    
    # There's clearly an issue with my recollection. Let me use a different verified source.
    # 
    # From a reliable reference, the 19 Laderman terms are:
    # (I'll use a formulation that I can verify numerically)
    
    # Let me just use the 189-rank approach (3x3 schoolbook blocks with Strassen 2x2)
    # which is known to work, and verify it.
    
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # For a 2x2 block A_block with entries [a00, a01; a10, a11]
    # and B_block with entries [b00, b01; b10, b11]:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # Flat indices for 2x2 blocks within 6x6 matrix
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    # Output C is p x n = 6 x 6, stored as W[k*n+i, r]
                    # So C[j, i] where j is col of C (which is row of A), i is row of C (which is col of B)
                    # Wait, let me re-read the convention: W[k*n+i, r]
                    # C[i,j] = sum_k A[i,k] B[k,j]
                    # W index: k*n + i, where k is the "column" of C (j in standard notation) and i is the "row" of C
                    # So W[j*n + i, t] corresponds to C[i, j]
                    
                    c00 = j * n + i  # C[i, j] top-left of 2x2 block
                    c01 = j * n + i + 1  # C[i, j+1]
                    c10 = (j+1) * n + i  # C[i+1, j]
                    c11 = (j+1) * n + i + 1  # C[i+1, j+1]
                    
                    if s == 0:
                        U[a00, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1
                        W[c11, idx] = 1
                    elif s == 1:
                        U[a10, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1
                        W[c11, idx] = -1
                    elif s == 2:
                        U[a00, idx] = 1
                        V[b01, idx] = 1
                        V[b11, idx] = -1
                        W[c01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 3:
                        U[a11, idx] = 1
                        V[b10, idx] = 1
                        V[b00, idx] = -1
                        W[c00, idx] = 1
                        W[c10, idx] = 1
                    elif s == 4:
                        U[a00, idx] = 1
                        U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1
                        W[c01, idx] = 1
                    elif s == 5:
                        U[a10, idx] = 1
                        U[a00, idx] = -1
                        V[b00, idx] = 1
                        V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:
                        U[a01, idx] = 1
                        U[a11, idx] = -1
                        V[b10, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verify correctness
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        # Efficient verification using matrix operations
        # C[i,j] = sum_t W[j*n+i, t] * (sum_k U[i*m+k, t] * V[k*p+j, t])
        
        # Build the full computation
        # For each term t, we have a rank-1 update
        C_computed = np.zeros((n, p))
        for t in range(rank):
            u_col = U_arr[:, t]  # shape (n*m,)
            v_col = V_arr[:, t]  # shape (m*p,)
            w_col = W_arr[:, t]  # shape (p*n,)
            
            # Reshape u_col to (n, m), v_col to (m, p)
            U_mat = u_col.reshape(n, m)
            V_mat = v_col.reshape(m, p)
            W_mat = w_col.reshape(p, n)
            
            # The contribution is: sum over k of U[i,k] * V[k,j] * W[j,i]
            # = (U @ V) element-wise multiplied by W, then... 
            # Actually: C[i,j] += W[j,i] * sum_k U[i,k] V[k,j]
            # = W^T * (U @ V) in some sense
            
            # Let me compute: M = U_mat @ V_mat, shape (n, p)
            # Then C[i,j] += W_mat[j, i] * M[i, j]
            M = U_mat @ V_mat  # (n, p)
            # W_mat is (p, n), so W_mat[j, i]
            # We need element-wise: C += W_mat.T * M  where W_mat.T is (n, p) with W_mat.T[i, j] = W_mat[j, i]
            C_computed += W_mat.T * M
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
