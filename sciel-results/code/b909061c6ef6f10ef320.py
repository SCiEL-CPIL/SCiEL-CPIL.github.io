import numpy as np
import time

def construct():
    _deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = 2, 5, 6
    
    # Rank calculation:
    # Split m=5 into blocks of size 2, 2, 1.
    # Block 1 (m=2): R(2, 2, 6) = 3 * R(2, 2, 2) = 3 * 7 = 21
    # Block 2 (m=2): R(2, 2, 6) = 3 * R(2, 2, 2) = 3 * 7 = 21
    # Block 3 (m=1): R(2, 1, 6) = 2 * 6 = 12 (Outer product)
    # Total Rank = 21 + 21 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen Coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U coeffs for A (i, j_loc)
    strassen_U = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (1,0): 1, (1,1): 1 }, # P2
        { (0,0): 1 },           # P3
        { (1,1): 1 },           # P4
        { (0,0): 1, (0,1): 1 }, # P5
        { (1,0): 1, (0,0): -1 },# P6
        { (0,1): 1, (1,1): -1 } # P7
    ]
    
    # V coeffs for B (j_loc, k_loc)
    strassen_V = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (0,0): 1 },           # P2
        { (0,1): 1, (1,1): -1 },# P3
        { (1,0): 1, (0,0): -1 },# P4
        { (1,1): 1 },           # P5
        { (0,0): 1, (0,1): 1 }, # P6
        { (1,0): 1, (1,1): 1 }  # P7
    ]
    
    # W coeffs for C (k_loc, i)
    # C11 (i=0, k=0): P1 + P4 - P5 + P7
    # C12 (i=0, k=1): P3 + P5
    # C21 (i=1, k=0): P2 + P4
    # C22 (i=1, k=1): P1 - P2 + P3 + P6
    
    strassen_W = [
        { (0,0): 1, (1,1): 1 }, # P1
        { (0,1): 1, (1,1): -1 },# P2
        { (1,0): 1, (1,1): 1 }, # P3
        { (0,0): 1, (0,1): 1 }, # P4
        { (0,0): -1, (1,0): 1 },# P5
        { (1,1): 1 },           # P6
        { (0,0): 1 }            # P7
    ]

    # Process Strassen Blocks for m=0..3 (J=0,1)
    for J in range(2):
        for K in range(3):
            for prod in range(7):
                r = col_idx
                col_idx += 1
                
                # Fill U
                for (i, j_loc), val in strassen_U[prod].items():
                    j_global = J * 2 + j_loc
                    U[i * m + j_global][r] += val
                
                # Fill V
                for (j_loc, k_loc), val in strassen_V[prod].items():
                    j_global = J * 2 + j_loc
                    k_global = K * 2 + k_loc
                    V[j_global * p + k_global][r] += val
                
                # Fill W
                for (k_loc, i), val in strassen_W[prod].items():
                    k_global = K * 2 + k_loc
                    W[k_global * n + i][r] += val

    # Process Outer Product Block for m=4 (J=3, j_global=4)
    j_global = 4
    for K in range(3):
        for i in range(2):
            for k_loc in range(2):
                r = col_idx
                col_idx += 1
                k_global = K * 2 + k_loc
                
                U[i * m + j_global][r] = 1
                V[j_global * p + k_global][r] = 1
                W[k_global * n + i][r] = 1

    if time.time() > _deadline:
        # Fallback: return a valid but potentially higher-rank decomposition
        # This shouldn't happen given the fixed construction
        pass
    return {"U": U, "V": V, "W": W}
