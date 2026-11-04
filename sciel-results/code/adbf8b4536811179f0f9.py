DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 50, then replace the 2x2x2 block with Strassen rank 7
    # Total rank = 50 - 8 + 7 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # M1: A[0,0]+A[1,1] * B[0,0]+B[1,1]
    # M2: A[1,0]+A[1,1] * B[0,0]
    # M3: A[0,0] * B[0,1]-B[1,1]
    # M4: A[1,1] * B[1,0]-B[0,0]
    # M5: A[0,0]+A[0,1] * B[1,1]
    # M6: A[1,0]-A[0,0] * B[0,0]+B[0,1]
    # M7: A[0,1]-A[1,1] * B[1,0]+B[1,1]
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Add Strassen terms (columns 0-6)
    for t, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
        for (i, j), val in u_coeffs.items():
            U[i * m + j][t] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][t] = val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][t] = val
    
    # Add schoolbook terms for the remaining 42 entries
    # Skip the 8 entries where i in {0,1}, j in {0,1}, k in {0,1}
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0,1) and j in (0,1) and k in (0,1):
                    continue  # Already handled by Strassen
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
