import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Block decomposition:
    # A is 4x5, B is 5x5
    # Split A into A1 (4x4) and a2 (4x1)
    # Split B into B1 (4x5) and b2 (1x5)
    # AB = A1*B1 + a2*b2
    # B1 = B1a (4x4) + b1b (4x1)
    # AB = A1*B1a + A1*b1b + a2*b2
    
    # Part 1: A1*B1a (4x4 * 4x4) using Strassen with schoolbook 2x2: rank 28
    # Part 2: A1*b1b (4x4 * 4x1) schoolbook: rank 16
    # Part 3: a2*b2 (4x1 * 1x5) schoolbook: rank 20
    # Total: 28 + 16 + 20 = 64
    
    rank = 28 + 16 + 20
    U = [[0]*rank for _ in range(n*m)]  # 20 x 64
    V = [[0]*rank for _ in range(m*p)]  # 25 x 64
    W = [[0]*rank for _ in range(p*n)]  # 20 x 64
    
    col = 0
    
    # Helper: add a schoolbook 2x2 matrix multiplication
    # u_terms: list of (U_row_idx, coeff) for the left 2x2 matrix
    # v_terms: list of (V_row_idx, coeff) for the right 2x2 matrix  
    # w_map: dict mapping (local_row, local_col) -> (W_row_idx, coeff)
    def add_schoolbook_2x2(u_terms, v_terms, w_map):
        nonlocal col
        for (u_i, c_u) in u_terms:
            for (v_j, c_v) in v_terms:
                for (lr, lc), (w_idx, c_w) in w_map.items():
                    U[u_i][col] += c_u
                    V[v_j][col] += c_v
                    W[w_idx][col] += c_w
                col += 1
        return col
    
    # Helper: add a schoolbook 2x1 matrix-vector multiplication
    # u_terms: list of (U_row_idx, coeff) for the 2x1 left vector
    # v_terms: list of (V_row_idx, coeff) for the 1x1 right scalar (just one entry)
    # w_map: dict mapping local_row -> (W_row_idx, coeff)
    def add_schoolbook_2x1(u_terms, v_terms, w_map):
        nonlocal col
        for (u_i, c_u) in u_terms:
            for (v_j, c_v) in v_terms:
                for lr, (w_idx, c_w) in w_map.items():
                    U[u_i][col] += c_u
                    V[v_j][col] += c_v
                    W[w_idx][col] += c_w
                col += 1
        return col
    
    # Helper: add a schoolbook 1x2 vector-matrix multiplication
    # u_terms: list of (U_row_idx, coeff) for the 1x1 left scalar
    # v_terms: list of (V_row_idx, coeff) for the 1x2 right vector
    # w_map: dict mapping local_col -> (W_row_idx, coeff)
    def add_schoolbook_1x2(u_terms, v_terms, w_map):
        nonlocal col
        for (u_i, c_u) in u_terms:
            for (v_j, c_v) in v_terms:
                for lc, (w_idx, c_w) in w_map.items():
                    U[u_i][col] += c_u
                    V[v_j][col] += c_v
                    W[w_idx][col] += c_w
                col += 1
        return col
    
    # PART 1: Strassen 4x4x4 decomposition
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # U indices for A: A[i,j] -> U[i*4+j], i,j in 0..3
    # V indices for B: B[j,k] -> V[j*4+k], j,k in 0..3
    # W indices for C: C[k,i] -> W[k*4+i], k,i in 0..3 (trace convention: W[k*n+i])
    
    # 2x2 block structure:
    # Block (r,c) of a 4x4 matrix covers rows 2r,2r+1 and cols 2c,2c+1
    # Local indices within block: (lr, lc) where lr,lc in {0,1}
    # Global row = 2*r + lr, global col = 2*c + lc
    
    # Strassen 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Mi, we need to specify the linear combinations in U and V,
    # then do schoolbook 2x2 multiplication, and map results to W.
    
    # Define the 7 Strassen multiplications
    # Each entry: (u_terms, v_terms, w_contributions)
    # u_terms: list of (U_row, coeff) for the left 2x2 matrix
    # v_terms: list of (V_row, coeff) for the right 2x2 matrix
    # w_contributions: list of (block_row, block_col, W_row, coeff) for each output block position
    
    # Block (r,c) of A: rows 2r,2r+1; cols 2c,2c+1
    # U index for A[2r+lr, 2c+lc] = (2r+lr)*4 + (2c+lc)
    
    # M1 = (A11+A22)(B11+B22)
    # A11: block (0,0), A22: block (1,1)
    # B11: block (0,0), B22: block (1,1)
    # Output goes to C11 (block 0,0) and C22 (block 1,1)
    m1_u = []
    for lr in range(2):
        for lc in range(2):
            # A11[lr,lc] = A[2*0+lr, 2*0+lc] = A[lr,lc]
            m1_u.append(((lr)*4 + lc, 1))
            # A22[lr,lc] = A[2*1+lr, 2*1+lc] = A[2+lr,2+lc]
            m1_u.append(((2+lr)*4 + (2+lc), 1))
    
    m1_v = []
    for lr in range(2):
        for lc in range(2):
            # B11[lr,lc] = B[lr,lc]
            m1_v.append((lr*4 + lc, 1))
            # B22[lr,lc] = B[2+lr,2+lc]
            m1_v.append(((2+lr)*4 + (2+lc), 1))
    
    m1_w = {}
    # C11 = M1 + ... : block (0,0), so C[2*0+lr, 2*0+lc] = C[lr,lc], W index = lc*4+lr
    for lr in range(2):
        for lc in range(2):
            m1_w[(lr, lc)] = (lc*4 + lr, 1)
    # C22 = M1 - ... : block (1,1), so C[2*1+lr, 2*1+lc] = C[2+lr,2+lc], W index = (2+lc)*4+(2+lr)
    for lr in range(2):
        for lc in range(2):
            key = (lr, lc)
            if key in m1_w:
                # Already added for C11, need separate entry for C22
                pass
            # We need to handle this differently - w_map should have unique keys
            # Let me restructure: w_map maps (block_r, block_c, lr, lc) -> (W_row, coeff)
    
    # I need to restructure. Let me use a different approach where w_map can have multiple entries
    # for the same local position but different blocks.
    
    # Let me redefine: w_map is a list of (block_r, block_c, lr, lc, W_row, coeff)
    # But that's messy. Instead, let me just directly add contributions.
    
    # Actually, let me simplify. For each Strassen multiplication, I'll directly
    # specify all the U, V, W contributions for the schoolbook 2x2 expansion.
    
    # For a 2x2 multiplication Xi * Yi = Zi:
    # Zi[lr,lc] = sum_lm Xi[lr,lm] * Yi[lm,lc]
    # Each term contributes: U[Xi_row] *= coeff_u, V[Yi_col] *= coeff_v, W[Z_row] += coeff_w
    
    # Let me define a helper that takes explicit U, V, W index lists with coefficients.
    
    def add_2x2_mult(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: list of (U_row, coeff) - 4 entries for the 2x2 left matrix
        v_coeffs: list of (V_row, coeff) - 4 entries for the 2x2 right matrix
        w_coeffs: list of (W_row, coeff) - 4 entries for the 2x2 result matrix
        This adds 4 rank-1 terms (schoolbook 2x2).
        """
        nonlocal col
        # u_coeffs[0] = (0,0), u_coeffs[1] = (0,1), u_coeffs[2] = (1,0), u_coeffs[3] = (1,1)
        # v_coeffs[0] = (0,0), v_coeffs[1] = (0,1), v_coeffs[2] = (1,0), v_coeffs[3] = (1,1)
        # w_coeffs[0] = (0,0), w_coeffs[1] = (0,1), w_coeffs[2] = (1,0), w_coeffs[3] = (1,1)
        
        # Schoolbook: 
        # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
        # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
        # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
        # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
        
        # Term 1: X[0,0]*Y[0,0] -> Z[0,0]
        U[u_coeffs[0][0]][col] += u_coeffs[0][1]
        V[v_coeffs[0][0]][col] += v_coeffs[0][1]
        W[w_coeffs[0][0]][col] += w_coeffs[0][1]
        col += 1
        
        # Term 2: X[0,1]*Y[1,0] -> Z[0,0]
        U[u_coeffs[1][0]][col] += u_coeffs[1][1]
        V[v_coeffs[2][0]][col] += v_coeffs[2][1]
        W[w_coeffs[0][0]][col] += w_coeffs[0][1]
        col += 1
        
        # Term 3: X[0,0]*Y[0,1] -> Z[0,1]
        U[u_coeffs[0][0]][col] += u_coeffs[0][1]
        V[v_coeffs[1][0]][col] += v_coeffs[1][1]
        W[w_coeffs[1][0]][col] += w_coeffs[1][1]
        col += 1
        
        # Term 4: X[0,1]*Y[1,1] -> Z[0,1]
        U[u_coeffs[1][0]][col] += u_coeffs[1][1]
        V[v_coeffs[3][0]][col] += v_coeffs[3][1]
        W[w_coeffs[1][0]][col] += w_coeffs[1][1]
        col += 1
        
        # Term 5: X[1,0]*Y[0,0] -> Z[1,0]
        U[u_coeffs[2][0]][col] += u_coeffs[2][1]
        V[v_coeffs[0][0]][col] += v_coeffs[0][1]
        W[w_coeffs[2][0]][col] += w_coeffs[2][1]
        col += 1
        
        # Term 6: X[1,1]*Y[1,0] -> Z[1,0]
        U[u_coeffs[3][0]][col] += u_coeffs[3][1]
        V[v_coeffs[2][0]][col] += v_coeffs[2][1]
        W[w_coeffs[2][0]][col] += w_coeffs[2][1]
        col += 1
        
        # Term 7: X[1,0]*Y[0,1] -> Z[1,1]
        U[u_coeffs[2][0]][col] += u_coeffs[2][1]
        V[v_coeffs[1][0]][col] += v_coeffs[1][1]
        W[w_coeffs[3][0]][col] += w_coeffs[3][1]
        col += 1
        
        # Term 8: X[1,1]*Y[1,1] -> Z[1,1]
        U[u_coeffs[3][0]][col] += u_coeffs[3][1]
        V[v_coeffs[3][0]][col] += v_coeffs[3][1]
        W[w_coeffs[3][0]][col] += w_coeffs[3][1]
        col += 1
        
        return col
    
    # Wait, that's 8 terms, but schoolbook 2x2 should be 4 terms.
    # I made an error. Schoolbook 2x2:
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]  (2 multiplications)
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]  (2 multiplications)
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]  (2 multiplications)
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]  (2 multiplications)
    # Total: 8 scalar multiplications? No, that's wrong.
    
    # Actually, for 2x2 matrix multiplication:
    # C = AB where A,B are 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # The tensor rank of 2x2x2 is 4. The schoolbook decomposition uses 4 rank-1 terms:
    # Term 1: A[0,0]*B[0,0] contributes to C[0,0]
    # Term 2: A[0,1]*B[1,0] contributes to C[0,0]
    # Term 3: A[0,0]*B[0,1] contributes to C[0,1]
    # Term 4: A[0,1]*B[1,1] contributes to C[0,1]
    # Term 5: A[1,0]*B[0,0] contributes to C[1,0]
    # Term 6: A[1,1]*B[1,0] contributes to C[1,0]
    # Term 7: A[1,0]*B[0,1] contributes to C[1,1]
    # Term 8: A[1,1]*B[1,1] contributes to C[1,1]
    
    # Wait, that's 8 terms. But the rank of 2x2x2 is 4, not 8.
    # I'm confusing myself. The "rank" in tensor decomposition means the number of
    # rank-1 tensors. For matrix multiplication A*B=C, each rank-1 term is of the form
    # u ⊗ v ⊗ w where u indexes A, v indexes B, w indexes C.
    
    # For schoolbook 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # This requires 2 rank-1 terms: (A[0,0], B[0,0], C[0,0]) and (A[0,1], B[1,0], C[0,0])
    # Similarly for other entries. Total: 8 rank-1 terms.
    
    # But wait, the known rank of 2x2x2 matrix multiplication is 4? No, I think I was wrong.
    # The rank of n×n×n matrix multiplication tensor is at most n^3 (schoolbook).
    # For n=2, schoolbook gives rank 8. Strassen shows it can be done with 7.
    # The exact rank of 2x2x2 is 7 (Strassen is optimal for 2x2).
    
    # So for 4x4x4:
    # Schoolbook: 4*4*4 = 64
    # Strassen (one level): 7 * (rank of 2x2x2) = 7 * 7 = 49
    # Or if we use schoolbook for the 2x2 blocks: 7 * 8 = 56
    
    # Let me just use schoolbook for everything to be safe. Rank = 64 for 4x4x4.
    # Then total = 64 + 16 + 20 = 100. That's the same as full schoolbook!
    
    # OK so the block decomposition doesn't help if I use schoolbook for the 4x4 core.
    # I need to actually implement Strassen properly.
    
    # Let me implement Strassen 2x2x2 with rank 7, then use it for the 4x4 core.
    # Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mi is a scalar multiplication (product of two linear combinations).
    # So each Mi contributes 1 rank-1 term.
    # Total: 7 rank-1 terms for 2x2x2.
    
    # For 4x4x4, I split into 2x2 blocks and use Strassen at the block level.
    # Each block-level "multiplication" is a 2x2x2 tensor with rank 7.
    # So total rank = 7 * 7 = 49.
    
    # This is getting complex. Let me just implement the full schoolbook for now
    # and verify it works, then optimize later.
    
    # Reset and do full schoolbook
    rank = n * m * p  # 100
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
