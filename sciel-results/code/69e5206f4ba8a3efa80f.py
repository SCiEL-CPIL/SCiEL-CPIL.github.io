"""
5x5 matrix multiplication - rank reduction via structured algebraic search.
Attempts to find a rank < 125 decomposition using Strassen-style recursive
block decomposition and randomized verification. Falls back to schoolbook if
no improvement is found within time limits.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    C[i,k] = sum_c W[k*n+i, c] * (A_flat . U[:,c]) * (B_flat . V[:,c])
    where A_flat has A[i,j] at position i*m+j, B_flat has B[j,k] at j*p+k.
    """
    r = U.shape[1]
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()  # length n*m, index i*m+j
    B_flat = B.flatten()  # length m*p, index j*p+k
    
    # Compute C from decomposition
    # L_c = A_flat . U[:,c] for each c
    L = A_flat @ U  # shape (r,)
    M = B_flat @ V  # shape (r,)
    
    # C[i,k] = sum_c W[k*n+i, c] * L[c] * M[c]
    # W is (p*n, r), W[k*n+i, c]
    # We need to reconstruct C from W, L, M
    # C_flat[k*n+i] = sum_c W[k*n+i, c] * L[c] * M[c]
    C_flat = W @ (L * M)  # shape (p*n,)
    C = C_flat.reshape(p, n).T  # C[i,k] = C_flat[k*n+i], so C_flat is (k,i) order
    
    # Actually let's be more careful.
    # C_flat has index k*n+i for element C[i,k]
    # So C_flat.reshape(p, n) gives [k, i]
    # We want C[i, k], so transpose
    C_computed = C_flat.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _strassen_2x2_decomposition():
    """
    Return the Strassen 2x2 decomposition as U, V, W factors.
    For 2x2 matrices:
    M1 = (A11-A22)(B12+B22)
    M2 = (A11+A22)B22
    M3 = A11(B11-B22)
    M4 = A22(B11+B22)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # n=m=p=2
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A11-A22)(B12+B22) = (A[0,0]-A[1,1])(B[0,1]+B[1,1])
    U[0, 0] = 1.0; U[3, 0] = -1.0
    V[1, 0] = 1.0; V[3, 0] = 1.0
    
    # M2 = (A11+A22)B22 = (A[0,0]+A[1,1])B[1,1]
    U[0, 1] = 1.0; U[3, 1] = 1.0
    V[3, 1] = 1.0
    
    # M3 = A11(B11-B22) = A[0,0](B[0,0]-B[1,1])
    U[0, 2] = 1.0
    V[0, 2] = 1.0; V[3, 2] = -1.0
    
    # M4 = A22(B11+B22) = A[1,1](B[0,0]+B[1,1])
    U[3, 3] = 1.0
    V[0, 3] = 1.0; V[3, 3] = 1.0
    
    # M5 = (A11+A12)B22 = (A[0,0]+A[0,1])B[1,1]
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    
    # M6 = (A21-A11)(B11+B12) = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    
    # M7 = (A12-A22)(B21+B22) = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    
    # W: C[i,k] coefficients
    # C[0,0] = M1+M4-M5+M7 -> W[0*2+0, c] = W[0, c]
    W[0, 0] = 1.0; W[0, 3] = 1.0; W[0, 4] = -1.0; W[0, 6] = 1.0
    
    # C[0,1] = M3+M5 -> W[1*2+0, c] = W[2, c]
    W[2, 2] = 1.0; W[2, 4] = 1.0
    
    # C[1,0] = M2+M4 -> W[0*2+1, c] = W[1, c]
    W[1, 1] = 1.0; W[1, 3] = 1.0
    
    # C[1,1] = M1-M2+M3+M6 -> W[1*2+1, c] = W[3, c]
    W[3, 0] = 1.0; W[3, 1] = -1.0; W[3, 2] = 1.0; W[3, 5] = 1.0
    
    return U, V, W

def _strassen_4x4_decomposition():
    """
    Return Strassen 4x4 decomposition with rank 49.
    Uses the 2x2 Strassen decomposition recursively.
    """
    n, m, p = 4, 4, 4
    r = 49
    
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # Get 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # For 4x4, we partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]] where each block is 2x2
    
    # Strassen for 4x4:
    # M1 = (A11-A22)(B12+B22) -> each term is 2x2, so rank 7 for each 2x2 mult
    # M2 = (A11+A22)B22
    # M3 = A11(B11-B22)
    # M4 = A22(B11+B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mk is a 2x2 matrix product, so uses rank 7
    # Total rank = 7 * 7 = 49
    
    # For each of the 7 Strassen terms, we need to construct
    # the linear combinations of A and B blocks, then apply
    # the 2x2 Strassen decomposition to each product.
    
    # Block indices in flat form:
    # A flat: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3,
    #         A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7,
    #         A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11,
    #         A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
    # 
    # Block A11: indices 0,1,4,5
    # Block A12: indices 2,3,6,7
    # Block A21: indices 8,9,12,13
    # Block A22: indices 10,11,14,15
    
    # Similarly for B and C
    
    # For each Strassen term k (0..6):
    for k in range(7):
        # Determine the linear combination coefficients for A and B
        # For term k, we have:
        # A_comb = c1*A11 + c2*A12 + c3*A21 + c4*A22
        # B_comb = d1*B11 + d2*B12 + d3*B21 + d4*B22
        
        if k == 0:  # M1 = (A11-A22)(B12+B22)
            a_coeffs = [1, 0, 0, -1]  # [A11, A12, A21, A22]
            b_coeffs = [0, 1, 0, 1]   # [B11, B12, B21, B22]
        elif k == 1:  # M2 = (A11+A22)B22
            a_coeffs = [1, 0, 0, 1]
            b_coeffs = [0, 0, 0, 1]
        elif k == 2:  # M3 = A11(B11-B22)
            a_coeffs = [1, 0, 0, 0]
            b_coeffs = [1, 0, 0, -1]
        elif k == 3:  # M4 = A22(B11+B22)
            a_coeffs = [0, 0, 0, 1]
            b_coeffs = [1, 0, 0, 1]
        elif k == 4:  # M5 = (A11+A12)B22
            a_coeffs = [1, 1, 0, 0]
            b_coeffs = [0, 0, 0, 1]
        elif k == 5:  # M6 = (A21-A11)(B11+B12)
            a_coeffs = [-1, 0, 1, 0]
            b_coeffs = [1, 1, 0, 0]
        elif k == 6:  # M7 = (A12-A22)(B21+B22)
            a_coeffs = [0, 1, 0, -1]
            b_coeffs = [0, 0, 1, 1]
        
        # Now for each of the 7 sub-columns in the 2x2 Strassen decomposition
        for sub_c in range(7):
            col = k * 7 + sub_c
            
            # U column: combination of A blocks, then 2x2 Strassen U
            # The 2x2 Strassen U acts on the 4-element flat vector of a 2x2 matrix
            # We need to map the 4-element result to the 16-element flat vector of 4x4
            
            # A block indices in 4x4 flat form
            # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
            a_block_indices = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
            
            # For each element in the 2x2 A_comb result
            for a_elem in range(4):
                # Which 2x2 Strassen U column? sub_c
                # What's the coefficient?
                u_coeff = U2[a_elem, sub_c]
                if u_coeff == 0:
                    continue
                
                # Which A blocks contribute to this element?
                # Element a_elem in the 2x2 block:
                # a_elem 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
                if a_elem == 0:  # top-left of 2x2 block
                    # All blocks have top-left at their respective positions
                    for b_idx, coeff in enumerate(a_coeffs):
                        if coeff != 0:
                            U[a_block_indices[b_idx][0], col] += coeff * u_coeff
                elif a_elem == 1:  # top-right
                    for b_idx, coeff in enumerate(a_coeffs):
                        if coeff != 0:
                            U[a_block_indices[b_idx][1], col] += coeff * u_coeff
                elif a_elem == 2:  # bottom-left
                    for b_idx, coeff in enumerate(a_coeffs):
                        if coeff != 0:
                            U[a_block_indices[b_idx][2], col] += coeff * u_coeff
                elif a_elem == 3:  # bottom-right
                    for b_idx, coeff in enumerate(a_coeffs):
                        if coeff != 0:
                            U[a_block_indices[b_idx][3], col] += coeff * u_coeff
            
            # V column: combination of B blocks, then 2x2 Strassen V
            b_block_indices = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
            
            for b_elem in range(4):
                v_coeff = V2[b_elem, sub_c]
                if v_coeff == 0:
                    continue
                
                if b_elem == 0:
                    for b_idx, coeff in enumerate(b_coeffs):
                        if coeff != 0:
                            V[b_block_indices[b_idx][0], col] += coeff * v_coeff
                elif b_elem == 1:
                    for b_idx, coeff in enumerate(b_coeffs):
                        if coeff != 0:
                            V[b_block_indices[b_idx][1], col] += coeff * v_coeff
                elif b_elem == 2:
                    for b_idx, coeff in enumerate(b_coeffs):
                        if coeff != 0:
                            V[b_block_indices[b_idx][2], col] += coeff * v_coeff
                elif b_elem == 3:
                    for b_idx, coeff in enumerate(b_coeffs):
                        if coeff != 0:
                            V[b_block_indices[b_idx][3], col] += coeff * v_coeff
            
            # W column: which C blocks this term contributes to
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            
            c_contrib = {}
            if k == 0:  # M1
                c_contrib = {0: 1, 3: 1}  # C11 and C22
            elif k == 1:  # M2
                c_contrib = {2: 1, 3: -1}  # C21 and C22
            elif k == 2:  # M3
                c_contrib = {1: 1, 3: 1}  # C12 and C22
            elif k == 3:  # M4
                c_contrib = {0: 1, 2: 1}  # C11 and C21
            elif k == 4:  # M5
                c_contrib = {0: -1, 1: 1}  # C11 and C12
            elif k == 5:  # M6
                c_contrib = {3: 1}  # C22
            elif k == 6:  # M7
                c_contrib = {0: 1}  # C11
            
            # C block indices in 4x4 flat form (C[k*n+i] convention)
            # C11: k in {0,1}, i in {0,1} -> indices 0,1,4,5
            # C12: k in {0,1}, i in {2,3} -> indices 2,3,6,7
            # C21: k in {2,3}, i in {0,1} -> indices 8,9,12,13
            # C22: k in {2,3}, i in {2,3} -> indices 10,11,14,15
            
            c_block_indices = {
                0: [0,1,4,5],   # C11
                1: [2,3,6,7],   # C12
                2: [8,9,12,13], # C21
                3: [10,11,14,15] # C22
            }
            
            for c_block, coeff in c_contrib.items():
                if coeff == 0:
                    continue
                for c_elem in range(4):
                    w_coeff = W2[c_elem, sub_c]
                    if w_coeff == 0:
                        continue
                    W[c_block_indices[c_block][c_elem], col] += coeff * w_coeff
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for 120s limit
    
    # First, get schoolbook as fallback
    U_s, V_s, W_s = _schoolbook(n, m, p)
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # Strategy 1: Use 4x4 Strassen (rank 49) + 1x1 cross terms
    # 5x5 = [4x4 block, 1x1 block] with cross terms
    # Partition:
    # A = [[A_4x4, A_4x1], [A_1x4, A_1x1]]
    # B = [[B_4x4, B_4x1], [B_1x4, B_1x1]]
    # C = [[C_4x4, C_4x1], [C_1x4, C_1x1]]
    
    # C_4x4 = A_4x4 * B_4x4 (rank 49 via Strassen)
    # C_4x1 = A_4x4 * B_1x1 + A_4x1 * B_1x1 -> A_4x4 * B_1x1 + A_4x1 * B_1x1
    #   = (A_4x4 + A_4x1_vec) * B_1x1 where we need to be careful
    #   Actually: C_4x1[i,0] = sum_j A[i,j] B[j,0] for i=0..3, j=0..3
    #   = sum_{j=0}^{3} A_4x4[i,j] B_4x1[j,0] + A_4x1[i,0] * B_1x1[0,0]
    #   Wait, let me re-index properly.
    
    # Let me use a cleaner approach. Partition as:
    # A is 5x5, partition into:
    #   Top-left 4x4: A[0:4, 0:4]
    #   Top-right 4x1: A[0:4, 4]
    #   Bottom-left 1x4: A[4, 0:4]
    #   Bottom-right 1x1: A[4, 4]
    
    # Similarly for B and C.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # For i < 4, k < 4: C_4x4[i,k] = sum_{j<4} A_4x4[i,j] B_4x4[j,k] + A_4x1[i] * B_1x4[k]
    #   Wait, j ranges 0..4. If j < 4: A[0:4,0:4][i,j] * B[0:4,0:4][j,k]
    #   If j = 4: A[0:4,4][i] * B[4,0:4][k]
    #   So C_4x4 = A_4x4 @ B_4x4 + A_4x1_vec * B_1x4_vec^T (outer product, rank 1)
    
    # For i < 4, k = 4: C_4x1[i] = sum_{j<4} A_4x4[i,j] B_4x1[j] + A_4x1[i] * B_1x1
    #   = (A_4x4 @ B_4x1)[i] + A_4x1[i] * B_1x1
    #   This is a matrix-vector product, can be done with rank 4*1 = 4 schoolbook + 1
    #   Actually A_4x4 @ B_4x1 is 4 mults per row, total 4*4 = 16 scalar mults
    #   Plus 4 more for A_4x1[i] * B_1x1, total 20
    
    # For i = 4, k < 4: C_1x4[k] = sum_{j<4} A_1x4[j] B_4x4[j,k] + A_1x1 * B_1x4[k]
    #   Similar: 4*4 + 4 = 20
    
    # For i = 4, k = 4: C_1x1 = sum_{j<4} A_1x4[j] B_4x1[j] + A_1x1 * B_1x1
    #   = 4 + 1 = 5 scalar mults
    
    # Total rank = 49 (Strassen 4x4) + 1 (outer product) + 16 + 4 + 16 + 4 + 4 + 1
    # Wait, that's not right. Let me count more carefully.
    
    # The cross terms:
    # 1. A_4x1_vec * B_1x4_vec^T: This is an outer product, rank 1
    #    C_4x4 += A_4x1 * B_1x4^T
    #    In tensor decomposition, this contributes rank 1
    
    # 2. A_4x4 @ B_4x1: This is 4x4 times 4x1, schoolbook rank 16
    # 3. A_4x1 * B_1x1: This is 4x1 times 1x1, schoolbook rank 4
    # 4. A_1x4 @ B_4x4: This is 1x4 times 4x4, schoolbook rank 16
    # 5. A_1x1 * B_1x4: This is 1x1 times 1x4, schoolbook rank 4
    # 6. A_1x4 @ B_4x1: This is 1x4 times 4x1, schoolbook rank 4
    # 7. A_1x1 * B_1x1: This is 1x1 times 1x1, rank 1
    
    # Total = 49 + 1 + 16 + 4 + 16 + 4 + 4 + 1 = 95
    
    # Let's construct this decomposition
    r_total = 95
    U = np.zeros((n*m, r_total), dtype=float)
    V = np.zeros((m*p, r_total), dtype=float)
    W = np.zeros((p*n, r_total), dtype=float)
    
    col = 0
    
    # Part 1: Strassen 4x4 for C_4x4 = A_4x4 @ B_4x4
    U4, V4, W4 = _strassen_4x4_decomposition()
    # U4 is (16, 49), V4 is (16, 49), W4 is (16, 49)
    # Map to 5x5 indices:
    # A_4x4 flat: A[0:4, 0:4] -> indices 0..15 in 5x5 flat (i*5+j for i<4,j<4)
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    #   A[1,0]=5, A[1,1]=6, A[1,2]=7, A[1,3]=8
    #   A[2,0]=10, A[2,1]=11, A[2,2]=12, A[2,3]=13
    #   A[3,0]=15, A[3,1]=16, A[3,2]=17, A[3,3]=18
    
    a_4x4_indices = [i*5+j for i in range(4) for j in range(4)]
    b_4x4_indices = [j*5+k for j in range(4) for k in range(4)]
    c_4x4_indices = [k*5+i for i in range(4) for k in range(4)]
    
    for c in range(49):
        for a_idx_2d, a_idx_1d in enumerate(a_4x4_indices):
            if U4[a_idx_2d, c] != 0:
                U[a_idx_1d, col+c] = U4[a_idx_2d, c]
        for b_idx_2d, b_idx_1d in enumerate(b_4x4_indices):
            if V4[b_idx_2d, c] != 0:
                V[b_idx_1d, col+c] = V4[b_idx_2d, c]
        for c_idx_2d, c_idx_1d in enumerate(c_4x4_indices):
            if W4[c_idx_2d, c] != 0:
                W[c_idx_1d, col+c] = W4[c_idx_2d, c]
    col += 49
    
    # Part 2: Outer product A_4x1 * B_1x4^T for C_4x4
    # C_4x4[i,k] += A[i,4] * B[4,k] for i,k < 4
    # This is rank 1
    # U: A[i,4] for i<4 -> A_flat[i*5+4]
    # V: B[4,k] for k<4 -> B_flat[4*5+k]
    # W: C[i,k] for i,k<4 -> W_flat[k*5+i]
    for i in range(4):
        U[i*5+4, col] = 1.0
    for k in range(4):
        V[4*5+k, col] = 1.0
    for i in range(4):
        for k in range(4):
            W[k*5+i, col] = 1.0
    col += 1
    
    # Part 3: A_4x4 @ B_4x1 for C_4x1
    # C[i,4] += sum_{j<4} A[i,j] B[j,4] for i<4
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            U[i*5+j, col] = 1.0
            V[j*5+4, col] = 1.0
            W[4*5+i, col] = 1.0
            col += 1
    
    # Part 4: A_4x1 * B_1x1 for C_4x1
    # C[i,4] += A[i,4] * B[4,4] for i<4
    # 4 terms
    for i in range(4):
        U[i*5+4, col] = 1.0
        V[4*5+4, col] = 1.0
        W[4*5+i, col] = 1.0
        col += 1
    
    # Part 5: A_1x4 @ B_4x4 for C_1x4
    # C[4,k] += sum_{j<4} A[4,j] B[j,k] for k<4
    # 16 terms
    for j in range(4):
        for k in range(4):
            U[4*5+j, col] = 1.0
            V[j*5+k, col] = 1.0
            W[k*5+4, col] = 1.0
            col += 1
    
    # Part 6: A_1x1 * B_1x4 for C_1x4
    # C[4,k] += A[4,4] * B[4,k] for k<4
    # 4 terms
    for k in range(4):
        U[4*5+4, col] = 1.0
        V[4*5+k, col] = 1.0
        W[k*5+4, col] = 1.0
        col += 1
    
    # Part 7: A_1x4 @ B_4x1 for C_1x1
    # C[4,4] += sum_{j<4} A[4,j] B[j,4]
    # 4 terms
    for j in range(4):
        U[4*5+j, col] = 1.0
        V[j*5+4, col] = 1.0
        W[4*5+4, col] = 1.0
        col += 1
    
    # Part 8: A_1x1 * B_1x1 for C_1x1
    U[4*5+4, col] = 1.0
    V[4*5+4, col] = 1.0
    W[4*5+4, col] = 1.0
    col += 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        best_rank = col
        best_U, best_V, best_W = U, V, W
    
    # If no improvement found, return schoolbook
    if best_rank == n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
