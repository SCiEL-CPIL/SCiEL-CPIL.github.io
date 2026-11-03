import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Block-Strassen decomposition for 6x6x6 matrix multiplication
    # Split into 3x3 blocks of 2x2 matrices
    # Each 2x2 block multiplication uses Strassen's 7-term algorithm
    # Total rank = 9 * 7 = 63... wait, that's not right.
    
    # Actually, let's think more carefully.
    # A is 6x6, split into 3x3 blocks of 2x2 each
    # B is 6x6, split into 3x3 blocks of 2x2 each
    # C[i,j] = sum_k A[i,k] * B[k,j] where each is 2x2 matrix product
    
    # For each pair of 2x2 blocks (A[i,k], B[k,j]), we need their 2x2 product
    # Strassen gives 7 scalar multiplications for 2x2 product
    # But we have 3*3*3 = 27 such block pairs, each needing 7 mults = 189
    
    # However, we can do better by combining across the 3x3 block structure
    # Let's use a simpler approach: apply Strassen at the 2x2 level within each block
    
    # Actually, let's just implement the straightforward block-Strassen:
    # For each of the 27 (i,k,j) triples, compute the 2x2 product using 7 terms
    # This gives rank 27*7 = 189
    
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block position (i, k, j) in the 3x3 grid:
    # A[i,k] is a 2x2 block at rows 2*i to 2*i+1, cols 2*k to 2*k+1
    # B[k,j] is a 2x2 block at rows 2*k to 2*k+1, cols 2*j to 2*j+1
    # C[i,j] is a 2x2 block at rows 2*i to 2*i+1, cols 2*j to 2*j+1
    
    # Strassen coefficients for each of the 7 terms:
    # Each term is a sum of products of entries from A-block and B-block
    
    # Let's define the Strassen terms more explicitly.
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
    
    # Now, for the full 6x6 case, we have 3x3 blocks.
    # For each (i, k, j), we compute 7 Strassen terms.
    # Each term involves specific entries of A[i,k] and B[k,j].
    
    # U encodes the A-side coefficients for each Strassen term
    # V encodes the B-side coefficients for each Strassen term
    # W encodes how each Strassen term contributes to C[i,j]
    
    # Let's index the 27 block positions and 7 Strassen terms
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # Define the Strassen term s for block (i,k,j)
                    # A block starts at row 2*i, col 2*k in A (flattened: (2*i)*6 + 2*k)
                    # B block starts at row 2*k, col 2*j in B (flattened: (2*k)*6 + 2*j)
                    # C block starts at row 2*i, col 2*j in C (flattened: (2*j)*6 + 2*i) per trace convention
                    
                    # A block entries (flattened indices in A's 6x6 = 36 elements)
                    # A[i,k] block: rows 2*i, 2*i+1; cols 2*k, 2*k+1
                    # Flattened: (2*i)*6 + 2*k, (2*i)*6 + 2*k+1, (2*i+1)*6 + 2*k, (2*i+1)*6 + 2*k+1
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    # B block entries (flattened indices in B's 6x6 = 36 elements)
                    # B[k,j] block: rows 2*k, 2*k+1; cols 2*j, 2*j+1
                    # Flattened: (2*k)*6 + 2*j, (2*k)*6 + 2*j+1, (2*k+1)*6 + 2*j, (2*k+1)*6 + 2*j+1
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    # C block entries (flattened indices in C's 6x6 = 36 elements, trace convention)
                    # C[i,j] block: rows 2*i, 2*i+1; cols 2*j, 2*j+1
                    # Trace convention: W[k*n+i, r] where k is col index, i is row index
                    # So C[r,c] is at W[c*6+r, r]
                    c00 = (2*j) * 6 + 2*i      # C[2*i, 2*j]
                    c01 = (2*j+1) * 6 + 2*i    # C[2*i, 2*j+1]
                    c10 = (2*j) * 6 + (2*i+1)  # C[2*i+1, 2*j]
                    c11 = (2*j+1) * 6 + (2*i+1)# C[2*i+1, 2*j+1]
                    
                    if s == 0:  # M1 = (a00 + a11)(b00 + b11)
                        U[a00, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1   # C00 += M1
                        W[c11, idx] = 1   # C11 += M1
                    elif s == 1:  # M2 = (a10 + a11)b00
                        U[a10, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1   # C10 += M2
                        W[c11, idx] = -1  # C11 -= M2
                    elif s == 2:  # M3 = a00(b01 - b11)
                        U[a00, idx] = 1
                        V[b01, idx] = 1
                        V[b11, idx] = -1
                        W[c01, idx] = 1   # C01 += M3
                        W[c11, idx] = 1   # C11 += M3
                    elif s == 3:  # M4 = a11(b10 - b00)
                        U[a11, idx] = 1
                        V[b10, idx] = 1
                        V[b00, idx] = -1
                        W[c00, idx] = 1   # C00 += M4
                        W[c10, idx] = 1   # C10 += M4
                    elif s == 4:  # M5 = (a00 + a01)b11
                        U[a00, idx] = 1
                        U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1  # C00 -= M5
                        W[c01, idx] = 1   # C01 += M5
                    elif s == 5:  # M6 = (a10 - a00)(b00 + b01)
                        U[a10, idx] = 1
                        U[a00, idx] = -1
                        V[b00, idx] = 1
                        V[b01, idx] = 1
                        W[c11, idx] = 1   # C11 += M6
                    elif s == 6:  # M7 = (a01 - a11)(b10 + b11)
                        U[a01, idx] = 1
                        U[a11, idx] = -1
                        V[b10, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1   # C00 += M7
                    
                    idx += 1
    
    # Verify we used all rank slots
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Quick sanity check: verify the decomposition is correct
    # C = A @ B should equal sum_r U[:,r] * V[:,r] outer product mapped through W
    # This is a spot check; for large matrices it could be slow, so we do a small check
    
    if time.time() < deadline - 5:
        # Create random test matrices
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        
        # Compute C = A @ B
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # The decomposition says: for each output position (r,c) of C,
        # C[r,c] = sum_k A[r,k] * B[k,c]
        # In the tensor decomposition: C[r,c] = sum_t U[r*m+k, t] * V[k*p+c, t] * W[c*n+r, t]
        # Wait, that's not quite right. Let me think about this more carefully.
        
        # The matrix multiplication tensor T has indices (i, k, j) where
        # T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j]
        # The decomposition is: T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[j,i,t]
        # where U is indexed by (i,k), V by (k,j), W by (j,i)
        
        # So C[i,j] = sum_k A[i,k] * B[k,j] = sum_k sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
        #            = sum_t W[j*n+i, t] * (sum_k U[i*m+k, t] * V[k*p+j, t])
        
        # Let's compute it this way
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for t in range(rank):
                    # sum_k U[i*m+k, t] * V[k*p+j, t]
                    s = 0.0
                    for k in range(m):
                        s += U_arr[i*m+k, t] * V_arr[k*p+j, t]
                    total += W_arr[j*n+i, t] * s
                C_computed[i, j] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook if Strassen fails
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
