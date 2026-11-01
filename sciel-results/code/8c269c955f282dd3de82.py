"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 14 (2 * 7)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # U coeffs for A (2x2)
    # V coeffs for B (2x2)
    # W coeffs for C (2x2)
    
    # Define the 7 Strassen terms for a single 2x2 block
    # Term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: [A00, A01, A10, A11]
    # v_coeffs: [B00, B01, B10, B11]
    # w_coeffs: [C00, C01, C10, C11]
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # Wait, M4 V is B10 - B00. B10 is idx 2, B00 is idx 0. So [0,0,1,-1] is correct? 
        # Let's re-verify M4.
        # M4 = A11 * (B10 - B00).
        # V coeffs: B00=1, B01=0, B10=1, B11=0? No.
        # B10 - B00.
        # B00 is at index 0. B10 is at index 2.
        # So V = [0, 0, 1, -1] is B10 - B00? No, index 2 is 1, index 0 is -1.
        # So V = [-1, 0, 1, 0].
        # Let's re-check M4 W.
        # C00 = M1 + M4 - M5 + M7. Coeff 1.
        # C10 = M2 + M4. Coeff 1.
        # C01, C11 coeff 0.
        # W = [1, 0, 1, 0]. Correct.
        # Let's re-check M2.
        # M2 = (A10 + A11) * B00.
        # U = [0, 0, 1, 1].
        # V = [1, 0, 0, 0].
        # C10 = M2 + M4. Coeff 1.
        # C11 = M1 - M2 + M3 + M6. Coeff -1.
        # W = [0, 0, 1, -1]. Correct.
        
        # Let's re-check M1.
        # M1 = (A00 + A11) * (B00 + B11).
        # U = [1, 0, 0, 1].
        # V = [1, 0, 0, 1].
        # C00 = M1 ... Coeff 1.
        # C11 = M1 ... Coeff 1.
        # W = [1, 0, 0, 1]. Correct.
        
        # Let's re-check M3.
        # M3 = A00 * (B01 - B11).
        # U = [1, 0, 0, 0].
        # V = [0, 1, 0, -1].
        # C01 = M3 + M5. Coeff 1.
        # C11 = ... + M3 ... Coeff 1.
        # W = [0, 1, 0, 1]. Correct.
        
        # Let's re-check M5.
        # M5 = (A00 + A01) * B11.
        # U = [1, 1, 0, 0].
        # V = [0, 0, 0, 1].
        # C00 = ... - M5 ... Coeff -1.
        # C01 = M3 + M5. Coeff 1.
        # W = [-1, 1, 0, 0]. Correct.
        
        # Let's re-check M6.
        # M6 = (A10 - A00) * (B00 + B01).
        # U = [-1, 0, 1, 0].
        # V = [1, 1, 0, 0].
        # C11 = ... + M6 ... Coeff 1.
        # W = [0, 0, 0, 1]. Correct.
        
        # Let's re-check M7.
        # M7 = (A01 - A11) * (B10 + B11).
        # U = [0, 1, 0, -1].
        # V = [0, 0, 1, 1].
        # C00 = ... + M7 ... Coeff 1.
        # W = [1, 0, 0, 0]. Correct.
    ]
    
    # Corrected M4 V: [-1, 0, 1, 0]
    strassen_terms[3] = ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0])
    
    # Fill first block (cols 0-6)
    for t in range(7):
        u_vals, v_vals, w_vals = strassen_terms[t]
        col = t
        
        # U: indices 0-3
        for i in range(4):
            U[i][col] = u_vals[i]
            
        # V: indices 0-3 (k=0,1)
        # V index mapping: j*2 + k
        # B00 (j=0,k=0) -> 0
        # B01 (j=0,k=1) -> 1
        # B10 (j=1,k=0) -> 2
        # B11 (j=1,k=1) -> 3
        for i in range(4):
            V[i][col] = v_vals[i]
            
        # W: indices 0,1,4,5 (k=0,1)
        # W index mapping: k*2 + i
        # C00 (k=0,i=0) -> 0
        # C01 (k=0,i=1) -> 1
        # C10 (k=1,i=0) -> 2? No.
        # Task says W[k*n+i]. n=2.
        # C_{i,k}.
        # k=0, i=0 -> 0*2+0 = 0.
        # k=0, i=1 -> 0*2+1 = 1.
        # k=1, i=0 -> 1*2+0 = 2.
        # k=1, i=1 -> 1*2+1 = 3.
        
        # Wait, my previous mapping was C10 -> 4?
        # Let's re-read the task description carefully.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W row index is $k \cdot n + i$.
        # For k=0: rows 0, 1. (C00, C01)
        # For k=1: rows 2, 3. (C10, C11)
        # For k=2: rows 4, 5. (C02, C03) -> Wait. C_{i,k}.
        # i is row of C. k is col of C.
        # C02: i=0, k=2. Index 2*2+0 = 4.
        # C03: i=0, k=3. Index 3*2+0 = 6.
        # C12: i=1, k=2. Index 2*2+1 = 5.
        # C13: i=1, k=3. Index 3*2+1 = 7.
        
        # So for first block (k=0,1):
        # C00 -> 0
        # C01 -> 1
        # C10 -> 2
        # C11 -> 3
        
        # My previous W mapping in strassen_terms was:
        # [C00, C01, C10, C11]
        # So:
        # W[0] = w_vals[0]
        # W[1] = w_vals[1]
        # W[2] = w_vals[2]
        # W[3] = w_vals[3]
        
        for i in range(4):
            W[i][col] = w_vals[i]

    # Fill second block (cols 7-13)
    for t in range(7):
        u_vals, v_vals, w_vals = strassen_terms[t]
        col = t + 7
        
        # U: same
        for i in range(4):
            U[i][col] = u_vals[i]
            
        # V: indices 4-7?
        # k=2,3.
        # B02 (j=0,k=2) -> 0*4+2 = 2.
        # B03 (j=0,k=3) -> 0*4+3 = 3.
        # B12 (j=1,k=2) -> 1*4+2 = 6.
        # B13 (j=1,k=3) -> 1*4+3 = 7.
        
        # v_vals is [B00, B01, B10, B11] for the local 2x2.
        # Local B00 corresponds to global B_{0,2} (idx 2).
        # Local B01 corresponds to global B_{0,3} (idx 3).
        # Local B10 corresponds to global B_{1,2} (idx 6).
        # Local B11 corresponds to global B_{1,3} (idx 7).
        
        V[2][col] = v_vals[0]
        V[3][col] = v_vals[1]
        V[6][col] = v_vals[2]
        V[7][col] = v_vals[3]
        
        # W: indices 4-7
        # Local C00 corresponds to global C_{0,2} (idx 4).
        # Local C01 corresponds to global C_{0,3} (idx 6).
        # Local C10 corresponds to global C_{1,2} (idx 5).
        # Local C11 corresponds to global C_{1,3} (idx 7).
        
        W[4][col] = w_vals[0]
        W[6][col] = w_vals[1]
        W[5][col] = w_vals[2]
        W[7][col] = w_vals[3]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
