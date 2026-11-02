import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A
    # P1: a11 + a22
    # P2: a21 + a22
    # P3: a11
    # P4: a22
    # P5: a11 + a12
    # P6: a21 - a11
    # P7: a12 - a22
    
    # Strassen coefficients for B (for a 2x2 block)
    # P1: b11 + b22
    # P2: b11
    # P3: b12 - b22
    # P4: b21 - b11
    # P5: b22
    # P6: b11 + b12
    # P7: b21 + b22
    
    # Strassen coefficients for C (for a 2x2 block)
    # P1: c11 += 1, c22 += -1
    # P2: c21 += 1, c22 += -1
    # P3: c12 += 1, c22 += -1
    # P4: c11 += 1, c21 += 1
    # P5: c11 += -1, c22 += 1
    # P6: c22 += 1
    # P7: c11 += 1, c12 += 1
    
    # Let's re-verify the C reconstruction
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 - P6
    
    # So for each P_s, we add its contribution to the C vector.
    
    A_coeffs = [
        [1, 0, 0, 1],  # P1: a11 + a22
        [0, 0, 1, 1],  # P2: a21 + a22
        [1, 0, 0, 0],  # P3: a11
        [0, 0, 0, 1],  # P4: a22
        [1, 1, 0, 0],  # P5: a11 + a12
        [-1, 0, 1, 0], # P6: a21 - a11
        [0, 1, 0, -1]  # P7: a12 - a22
    ]
    
    B_coeffs = [
        [1, 0, 0, 1],  # P1: b11 + b22
        [1, 0, 0, 0],  # P2: b11
        [0, 1, 0, -1], # P3: b12 - b22
        [0, 0, 1, -1], # P4: b21 - b11
        [0, 0, 0, 1],  # P5: b22
        [1, 1, 0, 0],  # P6: b11 + b12
        [0, 0, 1, 1]   # P7: b21 + b22
    ]
    
    C_coeffs = [
        [1, 0, 0, -1],  # P1: c11 + 1, c22 - 1
        [0, 0, 1, -1],  # P2: c21 + 1, c22 - 1
        [0, 1, 0, -1],  # P3: c12 + 1, c22 - 1
        [1, 0, 1, 0],   # P4: c11 + 1, c21 + 1
        [-1, 0, 0, 1],  # P5: c11 - 1, c22 + 1
        [0, 0, 0, 1],   # P6: c22 + 1
        [1, 1, 0, 0]    # P7: c11 + 1, c12 + 1
    ]
    
    # Map indices
    # A: i*m+j -> [0, 1, 2, 3] for a11, a12, a21, a22
    # B: j*p+k -> for block l, k=2l, 2l+1
    #    b11 is j=0, k=2l -> index 0*p + 2l
    #    b12 is j=0, k=2l+1 -> index 0*p + 2l+1
    #    b21 is j=1, k=2l -> index 1*p + 2l
    #    b22 is j=1, k=2l+1 -> index 1*p + 2l+1
    # C: k*n+i -> for block l, k=2l, 2l+1
    #    c11 is k=2l, i=0 -> index 2l*n + 0
    #    c12 is k=2l+1, i=0 -> index (2l+1)*n + 0
    #    c21 is k=2l, i=1 -> index 2l*n + 1
    #    c22 is k=2l+1, i=1 -> index (2l+1)*n + 1
    
    col = 0
    for l in range(6):
        for s in range(7):
            # Fill U
            for idx, val in enumerate(A_coeffs[s]):
                if val != 0:
                    U[idx][col] = val
            
            # Fill V
            # B_coeffs[s] is [b11, b12, b21, b22]
            # Indices:
            # b11: 0*p + 2l
            # b12: 0*p + 2l+1
            # b21: 1*p + 2l
            # b22: 1*p + 2l+1
            v_indices = [2*l, 2*l+1, p+2*l, p+2*l+1]
            for idx, val in enumerate(B_coeffs[s]):
                if val != 0:
                    V[v_indices[idx]][col] = val
            
            # Fill W
            # C_coeffs[s] is [c11, c12, c21, c22]
            # Indices:
            # c11: 2l*n + 0
            # c12: (2l+1)*n + 0
            # c21: 2l*n + 1
            # c22: (2l+1)*n + 1
            w_indices = [2*l*n, (2*l+1)*n, 2*l*n+1, (2*l+1)*n+1]
            for idx, val in enumerate(C_coeffs[s]):
                if val != 0:
                    W[w_indices[idx]][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
