import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    
    # Strategy:
    # The task asks for a rank-minimal exact tensor decomposition for C = A @ B.
    # A is 2x2, B is 2x16.
    # C is 2x16.
    # 
    # The tensor T has indices (i, j, k) where:
    # i in {0, 1} (row of A, row of C)
    # j in {0, 1} (col of A, row of B)
    # k in {0, ..., 15} (col of B, col of C)
    # 
    # Standard decomposition (schoolbook) uses sum_{i,j,k} e_{i,j}^A \otimes e_{j,k}^B \otimes e_{i,k}^C.
    # This has rank 2*2*16 = 64.
    #
    # However, we can view the multiplication as 4 independent multiplications of a 2x2 matrix A 
    # by a 2x4 submatrix of B? No, that's not quite right because A is shared.
    # 
    # Let's look at the structure.
    # C[:, k] = A @ B[:, k].
    # Let B be composed of 4 blocks of 4 columns: B_0, B_1, B_2, B_3, each 2x4.
    # C = A @ [B_0, B_1, B_2, B_3] = [A@B_0, A@B_1, A@B_2, A@B_3].
    # 
    # The tensor decomposition is linear.
    # If we can decompose the multiplication of a 2x2 matrix by a 2x4 matrix with rank R_2x2x4,
    # then the total rank for 2x2 by 2x16 would be 4 * R_2x2x4?
    # 
    # Is the rank of M_{2,2,4} less than 16?
    # The rank of M_{2,2,2} is 7 (Strassen).
    # M_{2,2,4} can be seen as two copies of M_{2,2,2}?
    # If we split B (2x4) into B_left (2x2) and B_right (2x2).
    # C_left = A @ B_left, C_right = A @ B_right.
    # The tensor for M_{2,2,4} is the direct sum of two M_{2,2,2} tensors?
    # Let's check the indices.
    # For M_{2,2,2}, indices are i, j in {0,1}, k in {0,1}.
    # For M_{2,2,4}, indices are i, j in {0,1}, k in {0,1,2,3}.
    # The tensor T_{i,j,k} = 1.
    # This is exactly two disjoint copies of the M_{2,2,2} tensor, one for k in {0,1} and one for k in {2,3}.
    # Since the supports in the C dimension (k) are disjoint, the tensor is a direct sum.
    # The rank of a direct sum of tensors is the sum of their ranks.
    # Rank(M_{2,2,4}) = Rank(M_{2,2,2}) + Rank(M_{2,2,2}) = 7 + 7 = 14.
    # 
    # Similarly, for M_{2,2,16}, we can split B into 4 blocks of 2x2.
    # B = [B_0, B_1, B_2, B_3] where each B_q is 2x2.
    # C = [C_0, C_1, C_2, C_3] where C_q = A @ B_q.
    # The tensor is the direct sum of 4 copies of M_{2,2,2}.
    # Therefore, Rank(M_{2,2,16}) = 4 * Rank(M_{2,2,2}) = 4 * 7 = 28.
    #
    # This is a significant improvement over 64.
    #
    # We need to construct the U, V, W matrices for this rank-28 decomposition.
    #
    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # We will use this structure for each of the 4 blocks.
    #
    # Total rank = 4 * 7 = 28.
    #
    # U matrix shape: (n*m, rank) = (4, 28)
    # V matrix shape: (m*p, rank) = (32, 28)
    # W matrix shape: (p*n, rank) = (32, 28)
    #
    # Indices:
    # A index: i*m + j. i in {0,1}, j in {0,1}.
    #   A[0,0] -> idx 0
    #   A[0,1] -> idx 1
    #   A[1,0] -> idx 2
    #   A[1,1] -> idx 3
    #
    # B index: j*p + k. j in {0,1}, k in {0..15}.
    #   B[0,0] -> idx 0
    #   B[0,1] -> idx 1
    #   ...
    #   B[0,15] -> idx 15
    #   B[1,0] -> idx 16
    #   ...
    #   B[1,15] -> idx 31
    #
    # C index: k*n + i. k in {0..15}, i in {0,1}.
    #   C[0,0] (k=0, i=0) -> idx 0
    #   C[0,1] (k=0, i=1) -> idx 1
    #   C[1,0] (k=1, i=0) -> idx 2
    #   C[1,1] (k=1, i=1) -> idx 3
    #   ...
    #   C[15,0] (k=15, i=0) -> idx 30
    #   C[15,1] (k=15, i=1) -> idx 31
    #
    # Let's define the Strassen coefficients.
    # We have 7 intermediate products M1..M7.
    # For each block q (0..3), the columns of B involved are k = 2*q, 2*q+1.
    # The rows of B are j=0 and j=1.
    # So B_block_q is:
    #   B[0, 2q], B[0, 2q+1]
    #   B[1, 2q], B[1, 2q+1]
    #
    # Let's map the Strassen inputs to the global B indices.
    # For block q:
    # B11_q = B[0, 2q]     -> idx 0*p + 2q = 2q
    # B12_q = B[0, 2q+1]   -> idx 0*p + 2q+1 = 2q+1
    # B21_q = B[1, 2q]     -> idx 1*p + 2q = 16 + 2q
    # B22_q = B[1, 2q+1]   -> idx 1*p + 2q+1 = 16 + 2q+1
    #
    # A indices are global and shared:
    # A11 = A[0,0] -> idx 0
    # A12 = A[0,1] -> idx 1
    # A21 = A[1,0] -> idx 2
    # A22 = A[1,1] -> idx 3
    #
    # C outputs for block q:
    # C11_q = C[0, 2q]     -> idx 2q*n + 0 = 4q
    # C12_q = C[0, 2q+1]   -> idx (2q+1)*n + 0 = 4q+2
    # C21_q = C[1, 2q]     -> idx 2q*n + 1 = 4q+1
    # C22_q = C[1, 2q+1]   -> idx (2q+1)*n + 1 = 4q+3
    #
    # Note on C index convention: W[k*n+i, r].
    # C[i, k] in standard math notation corresponds to element at row i, col k.
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]".
    # This usually means the index of the output element C[i,k] is k*n + i.
    # Let's verify:
    # If n=2, C[0,0] is i=0, k=0. Index 0*2+0=0.
    # C[0,1] is i=0, k=1. Index 1*2+0=2.
    # C[1,0] is i=1, k=0. Index 0*2+1=1.
    # C[1,1] is i=1, k=1. Index 1*2+1=3.
    #
    # So for block q:
    # C[0, 2q]   -> i=0, k=2q   -> idx 2q*2 + 0 = 4q
    # C[0, 2q+1] -> i=0, k=2q+1 -> idx (2q+1)*2 + 0 = 4q+2
    # C[1, 2q]   -> i=1, k=2q   -> idx 2q*2 + 1 = 4q+1
    # C[1, 2q+1] -> i=1, k=2q+1 -> idx (2q+1)*2 + 1 = 4q+3
    #
    # Now, construct U, V, W.
    # Rank R = 28.
    # We iterate over blocks q=0..3 and Strassen steps s=0..6.
    # Column index r = q*7 + s.
    
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen definitions
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Coefficients for U (A factors)
    # A11 idx 0, A12 idx 1, A21 idx 2, A22 idx 3
    # M1: A11(1), A22(1)
    # M2: A21(1), A22(1)
    # M3: A11(1)
    # M4: A22(1)
    # M5: A11(1), A12(1)
    # M6: A21(1), A11(-1)
    # M7: A12(1), A22(-1)
    
    # Coefficients for V (B factors)
    # B11 idx 2q, B12 idx 2q+1, B21 idx 16+2q, B22 idx 16+2q+1
    # M1: B11(1), B22(1)
    # M2: B11(1)
    # M3: B12(1), B22(-1)
    # M4: B21(1), B11(-1)
    # M5: B22(1)
    # M6: B11(1), B12(1)
    # M7: B21(1), B22(1)
    
    # Coefficients for W (C factors)
    # C11 idx 4q, C12 idx 4q+2, C21 idx 4q+1, C22 idx 4q+3
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the contribution of each M_s to the output C.
    # We need to fill W such that sum_s W[c_idx, s] * (U dot V) reproduces C.
    # Actually, the decomposition is T = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r].
    # This means C = sum_r (U[:,r] @ A) * (V[:,r] @ B) * W[:,r]? No.
    # The tensor is T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # And C[c] = sum_{a,b} T[a,b,c] A[a] B[b].
    # C[c] = sum_r W[c,r] (sum_a U[a,r] A[a]) (sum_b V[b,r] B[b]).
    # So W[c,r] is the coefficient for the product of the r-th linear form of A and the r-th linear form of B, contributing to output c.
    
    # For each block q and step s:
    # r = q*7 + s
    # U[:, r] corresponds to the A-factor of M_s
    # V[:, r] corresponds to the B-factor of M_s (specific to block q)
    # W[:, r] corresponds to the contribution of M_s to the outputs of block q
    
    # Let's populate U
    U_coeffs = {
        0: {0: 1, 3: 1},   # M1: A11, A22
        1: {2: 1, 3: 1},   # M2: A21, A22
        2: {0: 1},         # M3: A11
        3: {3: 1},         # M4: A22
        4: {0: 1, 1: 1},   # M5: A11, A12
        5: {2: 1, 0: -1},  # M6: A21, -A11
        6: {1: 1, 3: -1}   # M7: A12, -A22
    }
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
                
    # Let's populate V
    # V depends on q.
    # B11 idx 2q, B12 idx 2q+1, B21 idx 16+2q, B22 idx 16+2q+1
    V_coeffs = {
        0: {0: 1, 3: 1},   # M1: B11, B22
        1: {0: 1},         # M2: B11
        2: {1: 1, 3: -1},  # M3: B12, -B22
        3: {2: 1, 0: -1},  # M4: B21, -B11
        4: {3: 1},         # M5: B22
        5: {0: 1, 1: 1},   # M6: B11, B12
        6: {2: 1, 3: 1}    # M7: B21, B22
    }
    # Note: V_coeffs keys are local indices 0:B11, 1:B12, 2:B21, 3:B22
    # We need to map local indices to global B indices.
    # Global B index = j*p + k.
    # B11: j=0, k=2q -> idx 2q
    # B12: j=0, k=2q+1 -> idx 2q+1
    # B21: j=1, k=2q -> idx 16+2q
    # B22: j=1, k=2q+1 -> idx 16+2q+1
    
    local_to_global_B = [
        lambda q: 2*q,       # B11
        lambda q: 2*q + 1,   # B12
        lambda q: 16 + 2*q,  # B21
        lambda q: 16 + 2*q + 1 # B22
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for local_idx, coeff in V_coeffs[s].items():
                global_idx = local_to_global_B[local_idx](q)
                V[global_idx, r] = coeff
                
    # Let's populate W
    # W depends on q.
    # Output indices for block q:
    # C11: 4q
    # C12: 4q+2
    # C21: 4q+1
    # C22: 4q+3
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define W[c_idx, r] = coeff if M_s contributes coeff to C_c.
    # r = q*7 + s.
    
    # For a given q, the outputs are C11_q, C12_q, C21_q, C22_q.
    # Their global indices are 4q, 4q+2, 4q+1, 4q+3.
    
    # Contribution of M_s to each output:
    # M1 (s=0): C11(+1), C22(+1)
    # M2 (s=1): C21(+1), C22(-1)
    # M3 (s=2): C12(+1), C22(+1)
    # M4 (s=3): C11(+1), C21(+1)
    # M5 (s=4): C11(-1), C12(+1)
    # M6 (s=5): C22(+1)
    # M7 (s=6): C11(+1)
    
    # Let's encode this as a dictionary: s -> {output_local_idx: coeff}
    # output_local_idx: 0:C11, 1:C12, 2:C21, 3:C22
    W_contribs = {
        0: {0: 1, 3: 1},   # M1
        1: {2: 1, 3: -1},  # M2
        2: {1: 1, 3: 1},   # M3
        3: {0: 1, 2: 1},   # M4
        4: {0: -1, 1: 1},  # M5
        5: {3: 1},         # M6
        6: {0: 1}          # M7
    }
    
    # Map local output indices to global C indices for block q
    # C11 -> 4q
    # C12 -> 4q+2
    # C21 -> 4q+1
    # C22 -> 4q+3
    local_to_global_C = [
        lambda q: 4*q,     # C11
        lambda q: 4*q + 2, # C12
        lambda q: 4*q + 1, # C21
        lambda q: 4*q + 3  # C22
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for local_out_idx, coeff in W_contribs[s].items():
                global_out_idx = local_to_global_C[local_out_idx](q)
                W[global_out_idx, r] = coeff
                
    # Verification (optional, but good for safety)
    # Let's check if the decomposition is exact for a random instance.
    # C = A @ B
    # C[c] = sum_r W[c,r] * (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b])
    
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-10, 10, size=(n, m))
        B_test = np.random.randint(-10, 10, size=(m, p))
        C_test = A_test @ B_test
        
        # Flatten A and B
        A_flat = A_test.flatten() # i*m+j
        B_flat = B_test.flatten() # j*p+k
        
        # Compute C_decomp
        C_decomp = np.zeros(p * n, dtype=int)
        
        # For each r, compute the scalar product
        # term_r = (U[:,r] @ A_flat) * (V[:,r] @ B_flat)
        # Then C_decomp += W[:,r] * term_r
        
        # U, V, W are integer matrices.
        # We can do this vectorized.
        # U shape (4, 28), A_flat (4,) -> U.T @ A_flat (28,)
        # V shape (32, 28), B_flat (32,) -> V.T @ B_flat (28,)
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        
        C_decomp = W @ terms
        
        # C_test needs to be flattened in the same order as W indices.
        # W index c = k*n + i.
        # C_test is (n, p). C_test[i, k].
        # We need to rearrange C_test to match c = k*n + i.
        # C_test.T is (p, n). C_test.T.flatten() gives k*n + i.
        C_test_flat = C_test.T.flatten()
        
        if not np.allclose(C_decomp, C_test_flat):
            # If verification fails, fall back to schoolbook
            # This should not happen, but better safe than sorry.
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
