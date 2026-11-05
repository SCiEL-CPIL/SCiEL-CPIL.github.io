import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank 24 as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose T."""
        rank = U.shape[1]
        reconstructed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            reconstructed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, T, atol=tol)
    
    # Construct Rank 15 decomposition
    # 1 term for j=0
    # 7 terms for j in {1,2}, k in {0,1}
    # 7 terms for j in {1,2}, k in {2,3}
    
    rank_candidate = 15
    U = np.zeros((n*m, rank_candidate))
    V = np.zeros((m*p, rank_candidate))
    W = np.zeros((p*n, rank_candidate))
    
    r = 0
    
    # Term for j=0
    # A[:, 0] * B[0, :]
    # U[i*3+0, r] = 1 for i in {0,1}
    # V[0*4+k, r] = 1 for k in {0,1,2,3}
    # W[k*2+i, r] = 1
    for i in range(n):
        U[i*m + 0, r] = 1
    for k in range(p):
        V[0*p + k, r] = 1
        for i in range(n):
            W[k*n + i, r] = 1
    r += 1
    
    # Strassen blocks
    # Define Strassen coefficients for 2x2
    # A indices local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b11-b00)
    # m4 = a11*(b01-b00)
    # m5 = (a00+a10)*(b11+b10)
    # m6 = (a11-a10)*(b01+b00)
    # m7 = (a00-a11)*(b00-b11)
    
    # Coeffs for A part (length 4, indices 0,1,2,3)
    A_coeffs = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2 (a10+a11) -> local idx 2,3
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 0, 1, 0], # m5 (a00+a10) -> local idx 0,2
        [0, 0, -1, 1],# m6 (a11-a10) -> local idx 2,3
        [1, 0, 0, -1] # m7
    ]
    
    # Coeffs for B part (length 4, indices 0,1,2,3)
    B_coeffs = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 0, -1, 1],# m3 (b11-b00) -> local idx 2,3
        [0, 1, 0, -1],# m4 (b01-b00) -> local idx 1,0? No, b01 is idx 1, b00 is idx 0.
                      # Wait, local B idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
                      # b01 is B[0,1] -> local 1. b00 is B[0,0] -> local 0.
                      # So m4: b01 - b00 -> [0, 1, 0, -1] is wrong order?
                      # My local mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1).
                      # b01 is index 1. b00 is index 0.
                      # So coeff for idx 1 is 1, idx 0 is -1.
                      # Vector: [-1, 1, 0, 0].
                      # Let's re-verify Strassen formulas carefully.
        [0, 0, 1, 1], # m5 (b11+b10) -> b11 is idx 3, b10 is idx 2. [0,0,1,1]
        [1, 1, 0, 0], # m6 (b01+b00) -> b01 is idx 1, b00 is idx 0. [1,1,0,0]
        [1, 0, 0, -1] # m7 (b00-b11) -> b00 is idx 0, b11 is idx 3. [1,0,0,-1]
    ]
    
    # Correction for m3 and m4 B coeffs based on local indices:
    # m3: a00 * (b11 - b00). b11 is idx 3, b00 is idx 0. Coeffs: [ -1, 0, 0, 1 ]
    # m4: a11 * (b01 - b00). b01 is idx 1, b00 is idx 0. Coeffs: [ -1, 1, 0, 0 ]
    
    B_coeffs[2] = [-1, 0, 0, 1]
    B_coeffs[3] = [-1, 1, 0, 0]
    
    # Coeffs for C reconstruction (length 4, indices 0,1,2,3)
    # C00 = m1 + m4 - m5 - m7
    # C01 = m5 + m6
    # C10 = m2 + m3
    # C11 = m1 - m2 + m3 + m7
    # C local idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    C_coeffs = [
        [1, 0, 0, 0], # m1 contributes to C00
        [0, 0, 1, 0], # m2 contributes to C10
        [0, 0, 1, 0], # m3 contributes to C10
        [1, 0, 0, 0], # m4 contributes to C00
        [-1, 1, 0, 0],# m5 contributes to -C00 + C01
        [0, 1, 0, 0], # m6 contributes to C01
        [-1, 0, 0, 1] # m7 contributes to -C00 + C11
    ]
    
    # Wait, C_coeffs should be for each m_s, which C elements it contributes to.
    # m1: C00 (+1), C11 (+1)
    # m2: C10 (+1), C11 (-1)
    # m3: C10 (+1), C11 (+1)
    # m4: C00 (+1)
    # m5: C00 (-1), C01 (+1)
    # m6: C01 (+1)
    # m7: C00 (-1), C11 (+1)
    
    C_coeffs = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1],# m2
        [0, 0, 1, 1], # m3
        [1, 0, 0, 0], # m4
        [-1, 1, 0, 0],# m5
        [0, 1, 0, 0], # m6
        [-1, 0, 0, 1] # m7
    ]
    
    # Helper to add a Strassen block
    def add_strassen_block(j_start, k_start):
        nonlocal r
        for s in range(7):
            # A part
            for la in range(4):
                if A_coeffs[s][la] != 0:
                    # Local A index la: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # Global i, j
                    li = la // 2
                    lj = la % 2
                    gi = li
                    gj = j_start + lj
                    U[gi*m + gj, r] = A_coeffs[s][la]
            
            # B part
            for lb in range(4):
                if B_coeffs[s][lb] != 0:
                    # Local B index lb: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # Global j, k
                    lj = lb // 2
                    lk = lb % 2
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj*p + gk, r] = B_coeffs[s][lb]
            
            # C part
            for lc in range(4):
                if C_coeffs[s][lc] != 0:
                    # Local C index lc: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # Global k, i
                    li = lc // 2
                    lk = lc % 2
                    gi = li
                    gk = k_start + lk
                    W[gk*n + gi, r] = C_coeffs[s][lc]
            
            r += 1
    
    # Block 1: j in {1,2}, k in {0,1}
    add_strassen_block(1, 0)
    
    # Block 2: j in {1,2}, k in {2,3}
    add_strassen_block(1, 2)
    
    # Verify
    if verify(U, V, W):
        best_rank = rank_candidate
        best_result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return best_result
