import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen steps for 2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Coefficients for A parts
    # idx: 0:A11, 1:A12, 2:A21, 3:A22
    # Note: My U mapping for A1 was 0,1,4,5. 
    # Let's stick to local 0..3 for A1 and A2 separately in logic, then map to global.
    
    a_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    b_coeffs = [
        [1, 0, 1, 0], # M1 (B11, B12, B21, B22) -> 1,0,1,0? 
        # M1: (B11+B22). B11 is idx 0, B22 is idx 3. So [1,0,0,1]
        # Let's re-verify Strassen B coeffs.
        # M1: B11+B22 -> [1,0,0,1]
        # M2: B11 -> [1,0,0,0]
        # M3: B12-B22 -> [0,1,0,-1]
        # M4: B21-B11 -> [0,0,1,-1]
        # M5: B22 -> [0,0,0,1]
        # M6: B11+B12 -> [1,1,0,0]
        # M7: B21+B22 -> [0,0,1,1]
    ]
    
    b_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients
    # C11: M1 + M4 - M5 + M7
    # C12: M3 + M5
    # C21: M2 + M4
    # C22: M1 - M2 + M3 + M6
    # Local C indices: 0:C11, 1:C12, 2:C21, 3:C22
    w_coeffs = [
        [1, 0, 0, 1, 0, 0, 1],   # M1 contributes to C11, C22
        [0, 0, 1, 0, 0, -1, 0],  # M2 contributes to C21, C22(-)
        [0, 1, 0, 0, 0, 0, 1],   # M3 contributes to C12, C22
        [1, 0, 1, 0, 0, 0, 0],   # M4 contributes to C11, C21
        [-1, 1, 0, 0, 0, 0, 0],  # M5 contributes to C11(-), C12
        [0, 0, 0, 0, 0, 1, 0],   # M6 contributes to C22
        [1, 0, 0, 0, 0, 0, 0]    # M7 contributes to C11
    ]
    # Wait, let's re-verify W coeffs.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Col 0 (M1): C11(+1), C22(+1)
    # Col 1 (M2): C21(+1), C22(-1)
    # Col 2 (M3): C12(+1), C22(+1)
    # Col 3 (M4): C11(+1), C21(+1)
    # Col 4 (M5): C11(-1), C12(+1)
    # Col 5 (M6): C22(+1)
    # Col 6 (M7): C11(+1)
    
    w_coeffs = [
        [1, 0, 0, 1, 0, 0, 1],   # M1
        [0, 0, 1, 0, 0, -1, 0],  # M2
        [0, 1, 0, 0, 0, 0, 1],   # M3
        [1, 0, 1, 0, 0, 0, 0],   # M4
        [-1, 1, 0, 0, 0, 0, 0],  # M5
        [0, 0, 0, 0, 0, 1, 0],   # M6
        [1, 0, 0, 0, 0, 0, 0]    # M7
    ]
    
    # Global index mapping
    # A1: u 0,1,4,5 correspond to local 0,1,2,3
    # A2: u 2,3,6,7 correspond to local 0,1,2,3
    # B1k: v 2k, 2k+1, 6+2k, 7+2k correspond to local 0,1,2,3
    # B2k: v 12+2k, 13+2k, 18+2k, 19+2k correspond to local 0,1,2,3
    # Ck: w 4k, 4k+2, 4k+1, 4k+3 correspond to local 0,1,2,3
    
    col = 0
    for k in range(3):
        # Source 1: A1, B1k
        for s in range(7):
            # Fill U for A1
            for local_a in range(4):
                coeff = a_coeffs[s][local_a]
                if coeff != 0:
                    # Map local_a to global u for A1
                    if local_a == 0: u_idx = 0
                    elif local_a == 1: u_idx = 1
                    elif local_a == 2: u_idx = 4
                    elif local_a == 3: u_idx = 5
                    U[u_idx, col] = coeff
            
            # Fill V for B1k
            for local_b in range(4):
                coeff = b_coeffs[s][local_b]
                if coeff != 0:
                    # Map local_b to global v for B1k
                    if local_b == 0: v_idx = 2*k
                    elif local_b == 1: v_idx = 2*k + 1
                    elif local_b == 2: v_idx = 6 + 2*k
                    elif local_b == 3: v_idx = 7 + 2*k
                    V[v_idx, col] = coeff
            
            # Fill W for Ck
            for local_c in range(4):
                coeff = w_coeffs[s][local_c]
                if coeff != 0:
                    # Map local_c to global w for Ck
                    if local_c == 0: w_idx = 4*k
                    elif local_c == 1: w_idx = 4*k + 2
                    elif local_c == 2: w_idx = 4*k + 1
                    elif local_c == 3: w_idx = 4*k + 3
                    W[w_idx, col] = coeff
            
            col += 1
        
        # Source 2: A2, B2k
        for s in range(7):
            # Fill U for A2
            for local_a in range(4):
                coeff = a_coeffs[s][local_a]
                if coeff != 0:
                    # Map local_a to global u for A2
                    if local_a == 0: u_idx = 2
                    elif local_a == 1: u_idx = 3
                    elif local_a == 2: u_idx = 6
                    elif local_a == 3: u_idx = 7
                    U[u_idx, col] = coeff
            
            # Fill V for B2k
            for local_b in range(4):
                coeff = b_coeffs[s][local_b]
                if coeff != 0:
                    # Map local_b to global v for B2k
                    if local_b == 0: v_idx = 12 + 2*k
                    elif local_b == 1: v_idx = 13 + 2*k
                    elif local_b == 2: v_idx = 18 + 2*k
                    elif local_b == 3: v_idx = 19 + 2*k
                    V[v_idx, col] = coeff
            
            # Fill W for Ck
            for local_c in range(4):
                coeff = w_coeffs[s][local_c]
                if coeff != 0:
                    # Map local_c to global w for Ck
                    if local_c == 0: w_idx = 4*k
                    elif local_c == 1: w_idx = 4*k + 2
                    elif local_c == 2: w_idx = 4*k + 1
                    elif local_c == 3: w_idx = 4*k + 3
                    W[w_idx, col] = coeff
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
