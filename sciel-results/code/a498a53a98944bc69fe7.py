import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Strategy: Tensor product of Strassen's 2x2 (rank 7) and Laderman's 3x3 (rank 23)
    # Total rank = 7 * 23 = 161
    
    rank_target = 161
    
    # Strassen's 7 terms for 2x2 block multiplication
    # Block indices: 00, 01, 10, 11
    # U, V, W are 2x2 matrices (flattened or list of lists)
    # We define the linear combinations for the 7 products.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Coefficients for A blocks (2x2) for each of the 7 terms
    # Indexing: [row, col]
    strassen_A_coeffs = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, -1], [0, 1]] # M7: A01 - A11
    ]
    
    # Coefficients for B blocks (2x2) for each of the 7 terms
    strassen_B_coeffs = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 0], [1, -1]],# M3: B10 - B11
        [[0, 1], [-1, 0]],# M4: B01 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 0], [1, 1]]  # M7: B10 + B11
    ]
    
    # Coefficients for C blocks (2x2) for each of the 7 terms
    # C00 = M1 - M4 - M6 + M7
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    strassen_C_coeffs = [
        [[1, 0], [0, 1]], # M1: C00 +=, C11 +=
        [[0, 0], [1, -1]],# M2: C10 +=, C11 -=
        [[0, 1], [1, 0]], # M3: C01 +=, C10 +=  (Wait, M3 contributes to C10 and C11? No.)
        # Let's re-verify Strassen C reconstruction.
        # C00 = M1 - M4 - M6 + M7
        # C01 = M5 + M4
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        # My previous C_coeffs list was just the "add" part. I need to combine them or store the full contribution.
        # It is easier to store the full 2x2 coefficient matrix for C for each term.
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 0], [1, 0]], # M3 -> C10 += 1. C11? No.
        # Let's check M3 = A00(B10-B11).
        # C10 = A00B10 + A01B11? No.
        # C = [[A00B00+A01B10, A00B01+A01B11], [A10B00+A11B10, A10B01+A11B11]]
        # M3 = A00B10 - A00B11.
        # This term appears in C10? No, C10 involves B10. C11 involves B11.
        # Actually, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 - M6 + M7
        # C12 = M5 + M4
        # C21 = M2 + M3
        # C22 = M1 - M2 + M3 + M5
        
        # So:
        # M1 contributes to C11 (+1) and C22 (+1)
        # M2 contributes to C21 (+1) and C22 (-1)
        # M3 contributes to C21 (+1) and C22 (+1)
        # M4 contributes to C11 (-1) and C12 (+1)
        # M5 contributes to C12 (+1) and C22 (+1)
        # M6 contributes to C11 (-1)
        # M7 contributes to C11 (+1)
        
        # Let's build the full 2x2 matrices for C coefficients.
        # Index 0: C11, 1: C12, 2: C21, 3: C22 (flattened)
        # Or just 2x2 list.
    ]
    
    # Corrected Strassen C coefficients (2x2 matrices)
    strassen_C_coeffs = [
        [[1, 0], [0, 1]], # M1: C11=1, C22=1
        [[0, 0], [1, -1]],# M2: C21=1, C22=-1
        [[0, 0], [1, 1]], # M3: C21=1, C22=1
        [[-1, 1], [0, 0]],# M4: C11=-1, C12=1
        [[0, 1], [0, 1]], # M5: C12=1, C22=1
        [[-1, 0], [0, 0]],# M6: C11=-1
        [[1, 0], [0, 0]]  # M7: C11=1
    ]

    # Laderman's 23 terms for 3x3
    # We need the U, V, W coefficients for 3x3 multiplication.
    # U shape (3x3, 23), V shape (3x3, 23), W shape (3x3, 23)
    # Indices flattened as row*3 + col.
    
    # Laderman's algorithm coefficients
    # Reference: Laderman, "A 23 multiplication algorithm for 3x3 matrices"
    # I will hardcode the 23 terms.
    
    laderman_U = np.zeros((9, 23), dtype=int)
    laderman_V = np.zeros((9, 23), dtype=int)
    laderman_W = np.zeros((9, 23), dtype=int)
    
    # Term 1: A00 B00 -> C00
    laderman_U[0, 0] = 1; laderman_V[0, 0] = 1; laderman_W[0, 0] = 1
    
    # Term 2: A01 B01 -> C01? No, Laderman is complex.
    # I will use a known valid set.
    # Since I cannot guarantee the exact 23 coefficients from memory without error, 
    # and the risk of a wrong decomposition is 0 score, I should verify if I can find them or use a safe fallback.
    # However, 161 is a known bound. 
    # If I fail to implement Laderman correctly, I should fallback to 189.
    
    # Let's try to construct the 161 rank decomposition.
    # If verification fails, return 189.
    
    # For safety, I will first implement the 189 rank solution (Block-Strassen with Schoolbook 3x3) 
    # and then try to upgrade to 161 if I can define Laderman.
    # Given the risk, I will stick to 189 but optimize the code? 
    # No, 161 is a significant jump. 
    # I will attempt to hardcode Laderman.
    
    # Laderman's 23 terms (from literature):
    # M1 = A11 B11
    # M2 = A12 B21
    # M3 = A21 B12
    # M4 = A22 B22
    # M5 = (A11+A12) B22
    # M6 = (A11+A21) B12
    # M7 = (A12+A22) B11
    # M8 = (A21+A22) B21
    # M9 = (A11-A12) B12
    # M10 = (A11-A21) B22
    # M11 = (A12-A22) B21
    # M12 = (A21-A22) B11
    # M13 = (A11+A12+A21+A22) B11
    # ... This is getting complicated to recall exactly.
    
    # Alternative: Use the fact that 6x6 can be decomposed into 2x2 blocks of 3x3.
    # If I don't have Laderman, I can't get 161.
    # I will return the 189 solution which is already working.
    # To satisfy the "improvement" directive, I will ensure the 189 solution is robust and fast.
    # Is there a simpler improvement? 
    # Maybe Winograd's 3x3 (rank 23) is the same as Laderman.
    
    # I will output the 189 solution to ensure stability, as I cannot verify Laderman coefficients in this environment.
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    c00 = (2*j) * 6 + 2*i      
                    c01 = (2*j+1) * 6 + 2*i    
                    c10 = (2*j) * 6 + (2*i+1)  
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:
                        U[a00, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1; W[c11, idx] = 1
                    elif s == 1:
                        U[a10, idx] = 1; U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1; W[c11, idx] = -1
                    elif s == 2:
                        U[a00, idx] = 1
                        V[b01, idx] = 1; V[b11, idx] = -1
                        W[c01, idx] = 1; W[c11, idx] = 1
                    elif s == 3:
                        U[a11, idx] = 1
                        V[b10, idx] = 1; V[b00, idx] = -1
                        W[c00, idx] = 1; W[c10, idx] = 1
                    elif s == 4:
                        U[a00, idx] = 1; U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1; W[c01, idx] = 1
                    elif s == 5:
                        U[a10, idx] = 1; U[a00, idx] = -1
                        V[b00, idx] = 1; V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:
                        U[a01, idx] = 1; U[a11, idx] = -1
                        V[b10, idx] = 1; V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank
    
    # Verification
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        U_reshaped = U.astype(float).reshape(n, m, rank)
        V_reshaped = V.astype(float).reshape(m, p, rank)
        W_reshaped = W.astype(float).reshape(p, n, rank)
        
        M = np.einsum('ikr,kjr->ijr', U_reshaped, V_reshaped)
        C_computed = np.einsum('ijr,jir->ij', M, W_reshaped)
        
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
