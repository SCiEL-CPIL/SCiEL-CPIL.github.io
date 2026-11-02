"""Matrix multiplication 4,4,7 tensor decomposition - Kronecker/Strassen construction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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


def _build_strassen_2x2_2x2():
    """
    Strassen's rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_r U[alpha,r] * V[beta,r] * W[gamma,r] = delta_{target}
    
    Indexing: alpha = i*2+j for A[i,j], beta = j*2+k for B[j,k], gamma = k*2+i for C[i,k]
    Target: C[i,k] = sum_j A[i,j]*B[j,k]
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx2, A[1,1]=idx3
    # B indices: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx2, B[1,1]=idx3
    # C indices: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 -= 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11) -> C01 += 1, C11 += 1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00) -> C00 += 1, C10 += 1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_kronecker_4x4_4x7():
    """
    Build a rank-97 decomposition for 4x4 * 4x7 -> 4x7.
    
    Strategy: Split into two column-blocks of B:
      B = [B_left (4x4) | B_right (4x3)]
    
    For B_left: Use Kronecker product of Strassen (rank 7 for 2x2x2) with itself
    to get rank 7*7 = 49 for the 4x4 * 4x4 multiplication.
    
    For B_right: Use schoolbook for 4x4 * 4x3, which is 4*4*3 = 48 terms.
    
    Total rank: 49 + 48 = 97.
    
    Indexing convention:
      U: (n*m, rank) = (16, rank), indexed by [i*m + j] where i=row of A, j=col of A
      V: (m*p, rank) = (28, rank), indexed by [j*p + k] where j=row of B, k=col of B
      W: (p*n, rank) = (28, rank), indexed by [k*n + i] where k=col of C, i=row of C
    """
    n, m, p = 4, 4, 7
    
    # Build Strassen 2x2x2 components
    U_s, V_s, W_s = _build_strassen_2x2_2x2()
    
    # Kronecker product for 4x4 * 4x4 part
    # The 4x4 matrix A has blocks: A = [[A11, A12],[A21, A22]] where each is 2x2
    # Similarly B_left = [[B11, B12],[B21, B22]]
    # C_left = A * B_left
    #
    # For Kronecker: we need to map 4x4 indices to (2x2 outer) x (2x2 inner)
    # A[i,j] where i = i_out*2 + i_in, j = j_out*2 + j_in
    # The 4x4 multiplication can be seen as: for each pair of 2x2 block positions,
    # we do a 2x2 * 2x2 multiplication.
    #
    # Actually, the Kronecker product approach:
    # (A ⊗ I) doesn't directly work. Let me think more carefully.
    #
    # The 4x4 * 4x4 multiplication tensor is the tensor product of two 2x2 * 2x2 tensors
    # only if we use a specific block structure. Let's use the standard block decomposition:
    #
    # A = [[A_00, A_01], [A_10, A_11]]  (each 2x2)
    # B = [[B_00, B_01], [B_10, B_11]]  (each 2x2)
    # C = A*B where C_ab = sum_c A_ac * B_cb
    #
    # C_00 = A_00*B_00 + A_01*B_10
    # C_01 = A_00*B_01 + A_01*B_11
    # C_10 = A_10*B_00 + A_11*B_10
    # C_11 = A_10*B_01 + A_11*B_11
    #
    # Each C_ab is a 2x2 matrix computed from sums of 2x2*2x2 products.
    # This is NOT simply the tensor product of two Strassen algorithms.
    #
    # Instead, let's use a different approach: the 4x4x4 tensor rank is known to be <= 49
    # via the Kronecker product of two 2x2x2 tensors, but this requires a specific
    # "separable" structure. Let me use the direct Kronecker product construction.
    #
    # The tensor product T ⊗ T' for two bilinear algorithms gives a valid algorithm
    # for the tensor product of the corresponding multiplication problems.
    # 4x4 * 4x4 is NOT the tensor product of 2x2 * 2x2 with itself in the standard
    # index ordering. However, we can use the fact that:
    #
    # If we reorder the 4x4 indices as (i1,i2) where i = i1*2+i2, then the 4x4
    # multiplication is equivalent to two independent 2x2 multiplications in a
    # "block diagonal" sense only for special structures.
    #
    # Let me just use a simpler approach: apply Strassen at the block level.
    # 
    # Block-level Strassen for 4x4 * 4x4:
    # Split A, B into 2x2 blocks of 2x2 matrices.
    # Use Strassen's 7 multiplications at the block level, where each "multiplication"
    # is a 2x2 * 2x2 product computed with Strassen (rank 7).
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mx is a 2x2 * 2x2 product, computed with Strassen (7 scalar mults)
    # So we need 7 * 7 = 49 scalar multiplications total.
    
    # Let's build this explicitly.
    # For each of the 7 block-level Strassen products, we have a sum/difference of
    # 2x2 blocks. Each 2x2 block product uses Strassen's 7 products.
    
    # First, let's define the block-level Strassen coefficients.
    # For product M = X * Y where X and Y are 2x2 matrices (blocks):
    # The 7 Strassen products for X*Y are:
    # s1 = (X00+X11)(Y00+Y11)
    # s2 = (X10+X11)Y00
    # s3 = X00(Y10-Y11)
    # s4 = X11(Y01-Y00)
    # s5 = (X00+X01)Y11
    # s6 = (X10-X00)(Y00+Y01)
    # s7 = (X01-X11)(Y10+Y11)
    
    # Now, for the block-level Strassen, each "X" or "Y" is itself a sum/difference
    # of 2x2 blocks of A or B.
    
    # Let me build the full 49-term decomposition directly.
    
    # A indices: A[i,j] with i,j in 0..3. Block A[ab] where a,b in 0..1, 
    #   A[ab][i,j] = A[a*2+i, b*2+j]
    # B indices: B[j,k] with j,k in 0..3. Block B[cd] where c,d in 0..1,
    #   B[cd][i,j] = B[c*2+i, d*2+j]
    # C indices: C[i,k] with i,k in 0..3. Block C[ef] where e,f in 0..1,
    #   C[ef][i,j] = C[e*2+i, f*2+j]
    
    # The 49 scalar products: for each of 7 block-products, 7 Strassen products.
    # Let's enumerate them.
    
    # Block product s_block (b in 0..6) computes X_b * Y_b where:
    # X_b is a linear combination of A blocks, Y_b is a linear combination of B blocks.
    
    # Block-level Strassen definitions (X = A-blocks, Y = B-blocks):
    # b=0: X = A00+A11, Y = B00+B11
    # b=1: X = A10+A11, Y = B00
    # b=2: X = A00,       Y = B10-B11
    # b=3: X = A11,       Y = B01-B00
    # b=4: X = A00+A01,   Y = B11
    # b=5: X = A10-A00,   Y = B00+B01
    # b=6: X = A01-A11,   Y = B10+B11
    
    # For each block product b, we compute 7 Strassen sub-products.
    # Sub-product s (s in 0..6) for blocks X, Y:
    # s=0: (X00+X11)(Y00+Y11)
    # s=1: (X10+X11)Y00
    # s=2: X00(Y10-Y11)
    # s=3: X11(Y01-Y00)
    # s=4: (X00+X01)Y11
    # s=5: (X10-X00)(Y00+Y01)
    # s=6: (X01-X11)(Y10+Y11)
    
    # Let's build the U, V, W matrices for the 4x4*4x4 part (49 terms).
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))  # m*p for 4x4 is 16
    W_44 = np.zeros((16, 49))
    
    # Helper: for a 2x2 block at position (row_block, col_block) in a 4x4 matrix,
    # the 4 scalar entries are at indices:
    # (row_block*2 + 0, col_block*2 + 0) -> index row_block*2*4 + col_block*2 + 0 = 8*row_block + 2*col_block
    # (row_block*2 + 0, col_block*2 + 1) -> index 8*row_block + 2*col_block + 1
    # (row_block*2 + 1, col_block*2 + 0) -> index 8*row_block + 2*col_block + 2
    # (row_block*2 + 1, col_block*2 + 1) -> index 8*row_block + 2*col_block + 3
    
    def block_indices(row_block, col_block):
        """Return the 4 scalar indices for a 2x2 block in a 4x4 matrix."""
        base = 8 * row_block + 2 * col_block
        return [base, base + 1, base + 2, base + 3]
    
    # For U (A side): A[i,j], index = i*4+j
    # For V (B side): B[j,k], index = j*4+k (for 4x4 B)
    # For W (C side): C[i,k], index = k*4+i (per the convention W[k*n+i])
    
    # Strassen sub-product coefficients for a 2x2*2x2 multiplication:
    # For sub-product s, the X-coefficients for X00,X01,X10,X11 and Y-coefficients for Y00,Y01,Y10,Y11:
    
    strassen_x_coeffs = {
        0: [1, 0, 0, 1],   # X00+X11
        1: [0, 0, 1, 1],   # X10+X11
        2: [1, 0, 0, 0],   # X00
        3: [0, 0, 0, 1],   # X11
        4: [1, 1, 0, 0],   # X00+X01
        5: [-1, 0, 1, 0],  # X10-X00
        6: [0, 1, 0, -1],  # X01-X11
    }
    
    strassen_y_coeffs = {
        0: [1, 0, 0, 1],   # Y00+Y11
        1: [1, 0, 0, 0],   # Y00
        2: [0, 0, 1, -1],  # Y10-Y11
        3: [0, 1, 0, -1],  # Y01-Y00
        4: [0, 0, 0, 1],   # Y11
        5: [1, 1, 0, 0],   # Y00+Y01
        6: [0, 0, 1, 1],   # Y10+Y11
    }
    
    # Block-level Strassen: for block product b, X = sum_a alpha_b[a] * A_block[a],
    # Y = sum_c beta_b[c] * B_block[c]
    # where a,c index the 4 blocks (00,01,10,11)
    
    block_x_coeffs = {
        0: [1, 0, 0, 1],   # A00+A11
        1: [0, 0, 1, 1],   # A10+A11
        2: [1, 0, 0, 0],   # A00
        3: [0, 0, 0, 1],   # A11
        4: [1, 1, 0, 0],   # A00+A01
        5: [-1, 0, 1, 0],  # A10-A00
        6: [0, 1, 0, -1],  # A01-A11
    }
    
    block_y_coeffs = {
        0: [1, 0, 0, 1],   # B00+B11
        1: [1, 0, 0, 0],   # B00
        2: [0, 0, 1, -1],  # B10-B11
        3: [0, 1, 0, -1],  # B01-B00
        4: [0, 0, 0, 1],   # B11
        5: [1, 1, 0, 0],   # B00+B01
        6: [0, 0, 1, 1],   # B10+B11
    }
    
    # C output coefficients: C_block[e,f] = sum_b gamma_b[e,f] * M_b
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    c_coeffs = {
        (0, 0): [1, 0, 0, 1, -1, 0, 1],
        (0, 1): [0, 0, 1, 0, 1, 0, 0],
        (1, 0): [0, 1, 0, 1, 0, 0, 0],
        (1, 1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    # Now build the 49-term decomposition
    for b in range(7):  # block-level Strassen product
        for s in range(7):  # sub-Strassen product
            col = b * 7 + s
            
            # U coefficients: for each scalar in A
            # X = sum_a block_x_coeffs[b][a] * A_block[a]
            # Then X00 = sum_a block_x_coeffs[b][a] * A_block[a][0,0]
            # etc.
            # X_sub[s] = sum_{x_idx} strassen_x_coeffs[s][x_idx] * X[x_idx]
            # where x_idx: 0=X00, 1=X01, 2=X10, 3=X11
            
            for a in range(4):  # block index in A: 0=A00, 1=A01, 2=A10, 3=A11
                alpha = block_x_coeffs[b][a]
                if alpha == 0:
                    continue
                a_row_block = a // 2
                a_col_block = a % 2
                a_indices = block_indices(a_row_block, a_col_block)
                
                for x_idx in range(4):  # 0=X00, 1=X01, 2=X10, 3=X11
                    x_coeff = strassen_x_coeffs[s][x_idx]
                    if x_coeff == 0:
                        continue
                    # X[x_idx] corresponds to element [x_idx//2, x_idx%2] in the 2x2 block X
                    # X00 -> block element (0,0), X01 -> (0,1), X10 -> (1,0), X11 -> (1,1)
                    # A_block[a][x_idx//2, x_idx%2] = A[a_row_block*2 + x_idx//2, a_col_block*2 + x_idx%2]
                    a_i = a_row_block * 2 + x_idx // 2
                    a_j = a_col_block * 2 + x_idx % 2
                    a_scalar_idx = a_i * 4 + a_j
                    U_44[a_scalar_idx, col] += alpha * x_coeff
            
            # V coefficients: similar for B
            for c in range(4):  # block index in B: 0=B00, 1=B01, 2=B10, 3=B11
                beta = block_y_coeffs[b][c]
                if beta == 0:
                    continue
                b_row_block = c // 2
                b_col_block = c % 2
                
                for y_idx in range(4):  # 0=Y00, 1=Y01, 2=Y10, 3=Y11
                    y_coeff = strassen_y_coeffs[s][y_idx]
                    if y_coeff == 0:
                        continue
                    b_j = b_row_block * 2 + y_idx // 2
                    b_k = b_col_block * 2 + y_idx % 2
                    b_scalar_idx = b_j * 4 + b_k
                    V_44[b_scalar_idx, col] += beta * y_coeff
            
            # W coefficients: C_block[e,f] = sum_b c_coeffs[(e,f)][b] * M_b
            # M_b is a 2x2 matrix, and M_b[x_idx] contributes to C[e,f][x_idx]
            # C[e,f][x_idx] = C[e*2+x_idx//2, f*2+x_idx%2]
            for e in range(2):
                for f in range(2):
                    gamma = c_coeffs[(e, f)][b]
                    if gamma == 0:
                        continue
                    for x_idx in range(4):
                        # M_b[x_idx] contributes to C[e,f] at position (x_idx//2, x_idx%2)
                        # C[e,f][x_idx//2, x_idx%2] = C[e*2+x_idx//2, f*2+x_idx%2]
                        c_i = e * 2 + x_idx // 2
                        c_k = f * 2 + x_idx % 2
                        c_scalar_idx = c_k * 4 + c_i  # W convention: W[k*n+i] = W[k*4+i]
                        W_44[c_scalar_idx, col] += gamma
    
    # Now build the 4x4 * 4x3 schoolbook part (48 terms)
    U_43 = np.zeros((16, 48))
    V_43 = np.zeros((28, 48))  # m*p for 4x7 is 28, but we only use first 12 (4*3)
    W_43 = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):  # only 3 columns for B_right
                U_43[i * 4 + j, col] = 1
                V_43[j * 7 + (4 + k), col] = 1  # B[j, 4+k] for k=0,1,2
                W_43[(4 + k) * 4 + i, col] = 1  # C[i, 4+k]
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    U_full = np.hstack([U_44, U_43])
    V_full = np.hstack([V_44, V_43])
    W_full = np.hstack([W_44, W_43])
    
    return U_full, V_full, W_full


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n * m, W_mat.shape[1]):
        return False
    if V_mat.shape != (m * p, W_mat.shape[1]):
        return False
    if W_mat.shape != (p * n, U_mat.shape[1]):
        return False
    if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]:
        return False
    
    rank = U_mat.shape[1]
    
    # Verify target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Verify non-target entries are zero
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_set.add((i * m + j, j * p + k, k * n + i))
    
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                if (alpha, beta, gamma) not in target_set:
                    val = np.sum(U_mat[alpha, :] * V_mat[beta, :] * W_mat[gamma, :])
                    if abs(val) > tol:
                        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = U_school.shape[1]
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try the Kronecker/Strassen construction (rank 97)
    if time.time() < deadline - 10:
        U_k, V_k, W_k = _build_kronecker_4x4_4x7()
        
        if _verify(U_k, V_k, W_k, n, m, p, tol=1e-9):
            best_rank = U_k.shape[1]
            best_result = {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}
    
    # If we have time, try to search for even lower rank
    if best_rank < n * m * p and time.time() < deadline - 20:
        rng = np.random.RandomState(SEED)
        # Try a few ranks below 97
        for trial_rank in [96, 95, 94, 92, 90]:
            if time.time() > deadline - 15:
                break
            for restart in range(3):
                if time.time() > deadline - 15:
                    break
                U_try = rng.randn(n * m, trial_rank)
                V_try = rng.randn(m * p, trial_rank)
                W_try = rng.randn(p * n, trial_rank)
                
                # Quick gradient descent
                lr = 0.001
                for it in range(100):
                    if time.time() > deadline - 15:
                        break
                    dU = np.zeros_like(U_try)
                    dV = np.zeros_like(V_try)
                    dW = np.zeros_like(W_try)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                val = np.sum(U_try[u_idx, :] * V_try[v_idx, :] * W_try[w_idx, :])
                                diff = val - 1.0
                                dU[u_idx, :] += 2 * diff * V_try[v_idx, :] * W_try[w_idx, :]
                                dV[v_idx, :] += 2 * diff * U_try[u_idx, :] * W_try[w_idx, :]
                                dW[w_idx, :] += 2 * diff * U_try[u_idx, :] * V_try[v_idx, :]
                    
                    U_try -= lr * dU
                    V_try -= lr * dV
                    W_try -= lr * dW
                
                if _verify(U_try, V_try, W_try, n, m, p, tol=1e-8):
                    best_rank = trial_rank
                    best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                    break
    
    return best_result

# EVOLVE-BLOCK-END
