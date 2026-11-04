import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Strategy: Tensor product of Laderman's 3x3 (rank 23) and Strassen's 2x2 (rank 7)
    # Total rank = 23 * 7 = 161
    # If verification fails, fallback to Block-Strassen (rank 189)
    
    rank_target = 161
    
    # Define Strassen's 7 terms for 2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U, V, W are length 4 vectors
    strassen_U = [
        [1, 0, 0, 1], # M1: (a+d)
        [0, 0, 1, 1], # M2: (c+d)
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: (a+b)
        [-1, 0, 1, 0],# M6: (c-a)
        [0, 1, 0, -1] # M7: (b-d)
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: (e+h)
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: (f-h)
        [-1, 0, 1, 0],# M4: (g-e)
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: (e+f)
        [0, 0, 1, 1]  # M7: (g+h)
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+=, C11+=
        [0, 0, 1, -1],# M2: C10+=, C11-=
        [0, 1, 0, 1], # M3: C01+=, C11+=
        [1, 0, 1, 0], # M4: C00+=, C10+=
        [-1, 1, 0, 0],# M5: C00-=, C01+=
        [0, 0, 0, 1], # M6: C11+=
        [1, 0, 0, 0]  # M7: C00+=
    ]
    
    # Define Laderman's 23 terms for 3x3
    # Indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # This is the most critical part. If wrong, fallback triggers.
    # Using a known valid set of 23 terms.
    laderman_terms = [
        # (U, V, W) for 3x3 blocks
        # M1 = A11 B11
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        # M2 = (A11+A12) B22
        ([1,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]), # C22 += M2? No, C11?
        # Wait, C = AB. C_ij = sum A_ik B_kj.
        # M2 contributes to C_12? A11 B22 is not a term in schoolbook.
        # Laderman terms are bilinear forms that reconstruct C.
        # I need the exact W coefficients.
        # This is too risky to guess.
        
        # I will use a "safe" fallback to the current 189-rank implementation if I don't have the exact Laderman coefficients.
        # Since I don't have them verified in memory, I will stick to the current 189-rank implementation but clean it up.
        # However, I can try a simpler optimization: 
        # Is there a rank-180 decomposition?
        # No obvious one.
        
        # Let's just ensure the current 189-rank code is correct and fast.
        # The current code is already 189.
        
        # I will return the current implementation but with a slight optimization in verification.
    ]
    
    # Since I cannot guarantee Laderman coefficients, I will keep the 189-rank Block-Strassen.
    # I will just clean up the code and ensure the deadline is respected.
    
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
                    c11 = (2*j+1) * 6 + (2*i+1)#
                    
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
    
    # Verification using the official trace convention: W[k*n+i, r]
    # C[i,j] = sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) * W[j*n+i, r]
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        # U has shape (n*m, rank), indexed as U[i*m+k, r]
        # V has shape (m*p, rank), indexed as V[k*p+j, r]
        # W has shape (p*n, rank), indexed as W[j*n+i, r]
        
        # Compute C[i,j] = sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) * W[j*n+i, r]
        # First compute M[i,j,r] = sum_k U[i*m+k, r] * V[k*p+j, r]
        U_reshaped = U.astype(float).reshape(n, m, rank)  # U[i,k,r]
        V_reshaped = V.astype(float).reshape(m, p, rank)  # V[k,j,r]
        W_reshaped = W.astype(float).reshape(p, n, rank)  # W[j,i,r]
        
        # M[i,j,r] = sum_k U[i,k,r] * V[k,j,r]
        M = np.einsum('ikr,kjr->ijr', U_reshaped, V_reshaped)
        
        # C[i,j] = sum_r M[i,j,r] * W[j,i,r]
        C_computed = np.einsum('ijr,jir->ij', M, W_reshaped)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Try alternative W convention: W[i*n+j, r]
            W_reshaped_alt = W.astype(float).reshape(n, p, rank)  # W[i,j,r]
            C_computed_alt = np.einsum('ijr,ijr->ij', M, W_reshaped_alt)
            if np.allclose(C_computed_alt, C_expected, atol=1e-9):
                # The W matrix is using W[i*n+j, r] convention, need to transpose
                W_fixed = W_reshaped_alt.transpose(1, 0, 2).reshape(p * n, rank)
                return {"U": U.tolist(), "V": V.tolist(), "W": W_fixed.astype(int).tolist()}
            
            # Fallback to schoolbook with correct W convention W[j*n+i, r]
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
