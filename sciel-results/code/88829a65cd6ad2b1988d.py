import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # We will construct a decomposition based on Strassen's algorithm for the 4x4 core.
    # A is 4x5, B is 5x6.
    # Split A into A_core (4x4) and A_tail (4x1).
    # Split B into B_core (4x6) and B_tail (1x6).
    # C = A_core * B_core + A_tail * B_tail
    
    # Further split B_core into B_left (4x4) and B_right (4x2).
    # C = A_core * B_left + A_core * B_right + A_tail * B_tail
    
    # Term 1: A_core (4x4) * B_left (4x4). Use Strassen (rank 49).
    # Term 2: A_core (4x4) * B_right (4x2). Use Schoolbook (rank 32).
    # Term 3: A_tail (4x1) * B_tail (1x6). Use Schoolbook (rank 4).
    
    # Total Rank R = 49 + 32 + 4 = 85.
    
    R_strassen = 49
    R_right = 32
    R_tail = 4
    R_total = R_strassen + R_right + R_tail
    
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    # Indices for U (4x5), V (5x6), W (6x4)
    # U row index: i * 5 + j
    # V row index: j * 6 + k
    # W row index: k * 4 + i
    
    # --- Term 3: A_tail (4x1) * B_tail (1x6) ---
    # A_tail is col 4 of A. B_tail is row 4 of B.
    # Product contributes to all C[i, k].
    # T[i, j, k] = A[i, 4] * B[4, k]
    # This is a rank-4 tensor? No, it's sum of 4 rank-1 terms?
    # A[i, 4] * B[4, k] = sum_{j=0}^{4} A[i, 4] * delta_{j,4} * B[4, k]
    # Wait, the tensor is T[i, j, k].
    # For this term, j must be 4.
    # So T[i, 4, k] = A[i, 4] * B[4, k].
    # This can be decomposed into 4 rank-1 tensors?
    # A[i, 4] * B[4, k] = A[i, 4] * 1 * B[4, k].
    # We need to express this as sum_r U[i, 4, r] V[4, k, r] W[k, i, r].
    # Since j is fixed at 4, V[4, k, r] is the only V entry involved for that j.
    # We can set U[i, 4, r] = A[i, 4] (but we don't know A, we need a basis).
    # The decomposition must be exact for ANY A, B.
    # So we need basis vectors.
    # For the term A[i, 4] B[4, k], we can use 4 basis vectors corresponding to i=0..3?
    # No, A[i, 4] is a vector of length 4. B[4, k] is a vector of length 6.
    # The outer product is a 4x6 matrix.
    # We can decompose it into 4 rank-1 matrices:
    # E_{i, k} for i=0..3, k=0..5? No, that's 24.
    # Actually, A[i, 4] B[4, k] is the (i, k) entry of the outer product.
    # We can write A[i, 4] B[4, k] = sum_{s=0}^{3} A[i, 4] delta_{s, i} * 1 * B[4, k] delta_{s, i}? No.
    # Let's use the standard basis.
    # Let r be 0..3.
    # U[i, 4, r] = delta_{i, r}
    # V[4, k, r] = 1 (for all k, r? No, V depends on k)
    # If V[4, k, r] = 1, then sum_r delta_{i, r} * 1 * W[k, i, r] = W[k, i, i].
    # We want this to be A[i, 4] B[4, k].
    # This doesn't work directly because U and V are fixed basis matrices.
    # The decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # For j=4, we want T[i, 4, k] = A[i, 4] B[4, k].
    # This term is linear in A and B.
    # We can set:
    # For r in 0..3:
    #   U[i, 4, r] = delta_{i, r}
    #   V[4, k, r] = 1
    #   W[k, i, r] = 0? No.
    #   If W[k, i, r] = 0, the term is 0.
    #   We need W to depend on A and B? No, U, V, W are fixed.
    #   Wait, the decomposition is for the TENSOR of multiplication.
    #   T[i, j, k] is the coefficient of A[i, j] B[j, k] in the product C[i, k].
    #   T[i, j, k] = 1 if the product contributes to C[i, k], else 0.
    #   For the term A[i, 4] B[4, k], the contribution is to C[i, k].
    #   So T[i, 4, k] = 1.
    #   We need to represent the tensor T_tail where T_tail[i, 4, k] = 1 and 0 otherwise.
    #   This tensor has rank 4?
    #   T_tail = sum_{i=0}^3 sum_{k=0}^5 e_i \otimes e_4 \otimes e_k? No.
    #   T_tail[i, j, k] = delta_{j, 4} delta_{i, i} delta_{k, k}?
    #   T_tail = sum_{i=0}^3 sum_{k=0}^5 u_i \otimes v_4 \otimes w_{k, i}
    #   We can choose u_i = e_i (in R^4), v_4 = e_4 (in R^5), w_{k, i} = e_k \otimes e_i (in R^{24})?
    #   No, W is p x n = 6 x 4.
    #   The tensor is T[i, j, k].
    #   We can decompose T_tail into 4 terms?
    #   T_tail[i, j, k] = delta_{j, 4} * (sum_{s=0}^3 delta_{i, s} delta_{k, s}?) No.
    #   T_tail[i, 4, k] = 1 for all i, k.
    #   This is the all-ones matrix in the (i, k) plane, at j=4.
    #   Rank of all-ones 4x6 matrix is 1.
    #   So T_tail can be represented with rank 1?
    #   T_tail[i, j, k] = delta_{j, 4} * 1 * 1.
    #   U[i, 4, r] = 1
    #   V[4, k, r] = 1
    #   W[k, i, r] = 1
    #   Then sum_r U V W = 1 * 1 * 1 = 1.
    #   So R_tail = 1 is sufficient!
    #   Let's check:
    #   U[i, 4, 0] = 1
    #   V[4, k, 0] = 1
    #   W[k, i, 0] = 1
    #   T[i, 4, k] = 1. Correct.
    #   T[i, j, k] = 0 for j != 4. Correct.
    
    R_tail = 1
    R_total = R_strassen + R_right + R_tail
    
    # Re-initialize
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    # --- Term 3: A_tail * B_tail (Rank 1) ---
    # r = 0
    for i in range(n):
        U[i * m + 4, 0] = 1
    for k in range(p):
        V[4 * p + k, 0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, 0] = 1
            
    # --- Term 2: A_core * B_right (4x4 * 4x2) ---
    # A_core is A[0:4, 0:4]. B_right is B[0:4, 4:6].
    # T[i, j, k] = 1 if i<4, j<4, k>=4.
    # Rank is 4*4*2 = 32.
    r_offset = 1
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j, r_offset] = 1
                V[j * p + k, r_offset] = 1
                W[k * n + i, r_offset] = 1
                r_offset += 1
                
    # --- Term 1: A_core * B_left (4x4 * 4x4) ---
    # Use Strassen's algorithm for 4x4.
    # Strassen for 4x4 uses 7 multiplications of 2x2 blocks.
    # We need to implement the Strassen decomposition for 4x4x4.
    # The rank is 49.
    # We will generate the 49 rank-1 terms.
    
    r_offset = 1 + 32
    # Strassen for 2x2 has 7 terms.
    # For 4x4, we split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
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
    
    # Each M is a 2x2 matrix product.
    # We can use the schoolbook for the 2x2 products inside Strassen?
    # No, Strassen replaces the 8 products with 7.
    # But each "product" here is a 2x2 matrix multiplication.
    # So we need a decomposition for 2x2x2 with rank 8 (schoolbook) or 7 (Strassen)?
    # If we use schoolbook for the 2x2 sub-products, the total rank is 7 * 8 = 56.
    # If we use Strassen for the 2x2 sub-products, the total rank is 7 * 7 = 49.
    # Let's use Strassen recursively? Or just schoolbook for 2x2?
    # Schoolbook 2x2 rank is 8. 7*8 = 56.
    # Strassen 2x2 rank is 7. 7*7 = 49.
    # Let's implement the 2x2 Strassen decomposition.
    
    # 2x2 Strassen Terms (7 terms)
    # U, V, W for 2x2x2.
    # n=2, m=2, p=2.
    # T[i, j, k] = 1.
    # M1: (A11+A22)(B11+B22) -> C11, C22
    # ...
    
    # Let's define the 7 Strassen tensors for 2x2.
    # Each tensor is a sum of rank-1 terms? No, each Strassen step is a product of sums.
    # We need to expand the sums into rank-1 terms.
    
    # Strassen 2x2 Decomposition (7 rank-8? No, 7 terms, each term is a product of 2x2 matrices).
    # Each product of 2x2 matrices has rank 8 (schoolbook).
    # So total rank 56.
    # To get rank 49, we need to use Strassen for the 2x2 products too.
    # This is getting complex.
    
    # Let's just use Schoolbook for the 4x4x4 core?
    # Rank 64.
    # Total Rank = 64 + 32 + 1 = 97.
    # This is still better than 120.
    # And much easier to implement correctly.
    
    R_strassen = 64 # Using schoolbook for simplicity and correctness
    R_total = R_strassen + R_right + R_tail
    
    # Re-initialize
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    # --- Term 3: A_tail * B_tail (Rank 1) ---
    r_offset = 0
    for i in range(n):
        U[i * m + 4, r_offset] = 1
    for k in range(p):
        V[4 * p + k, r_offset] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r_offset] = 1
    r_offset += 1
    
    # --- Term 2: A_core * B_right (4x4 * 4x2) ---
    # Rank 32
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j, r_offset] = 1
                V[j * p + k, r_offset] = 1
                W[k * n + i, r_offset] = 1
                r_offset += 1
                
    # --- Term 1: A_core * B_left (4x4 * 4x4) ---
    # Rank 64 (Schoolbook)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j, r_offset] = 1
                V[j * p + k, r_offset] = 1
                W[k * n + i, r_offset] = 1
                r_offset += 1
                
    # Verify
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kr->ijk', U_t, V_t, W_t)
        # T[i, j, k] should be 1 for all i, j, k
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if construction fails (should not happen)
    R_sb = n * m * p
    U_sb = np.zeros((n * m, R_sb))
    V_sb = np.zeros((m * p, R_sb))
    W_sb = np.zeros((p * n, R_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
