import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # a00, a01, a10, a11
    # Indices in U: i*m + j -> 0:00, 1:01, 2:10, 3:11
    
    # Strassen linear forms for B block (2x2)
    # b00, b01, b10, b11
    # Indices in V: j*p + k
    
    # Coefficients for C (2x2)
    # C00, C01, C10, C11
    # Indices in W: k*n + i -> 0:00, 1:01, 2:10, 3:11
    
    # s: 0..6
    # u_coeffs: dict mapping (i,j) to coeff
    # v_coeffs: dict mapping (j,k) to coeff
    # w_coeffs: dict mapping (k,i) to coeff
    
    strassen_u = [
        {(0,0): 1, (1,1): 1}, # M1: a00+a11
        {(1,0): 1, (1,1): 1}, # M2: a10+a11
        {(0,0): 1},           # M3: a00
        {(1,1): 1},           # M4: a11
        {(0,0): 1, (0,1): 1}, # M5: a00+a01
        {(1,0): 1, (0,0): -1},# M6: a10-a00
        {(0,1): 1, (1,1): -1} # M7: a01-a11
    ]
    
    # v_coeffs for a generic block B_t
    # b00, b01, b10, b11
    strassen_v = [
        {(0,0): 1, (1,1): 1}, # x+w
        {(0,0): 1},           # x
        {(0,1): 1, (1,1): -1},# y-w
        {(1,0): 1, (0,0): -1},# z-x
        {(1,1): 1},           # w
        {(0,0): 1, (0,1): 1}, # x+y
        {(1,0): 1, (1,1): 1}  # z+w
    ]
    
    # w_coeffs for C
    # C00, C01, C10, C11
    # (k,i) indices: 00, 01, 10, 11
    strassen_w = [
        {(0,0): 1, (1,1): 1}, # M1
        {(0,1): 1},           # M2
        {(1,1): -1, (1,0): 1},# M3
        {(0,0): -1, (0,1): 1},# M4
        {(0,0): 1, (1,0): 1}, # M5
        {(1,1): 1},           # M6
        {(0,0): -1, (1,1): 1} # M7
    ]
    
    for t in range(3):
        for s in range(7):
            c = 7 * t + s
            # Fill U
            for (i, j), val in strassen_u[s].items():
                U[i * m + j][c] += val
            # Fill V
            for (j, k), val in strassen_v[s].items():
                V[j * p + (2 * t + k)][c] += val
            # Fill W
            for (k, i), val in strassen_w[s].items():
                W[(2 * t + k) * n + i][c] += val
                
    return {"U": U, "V": V, "W": W}
