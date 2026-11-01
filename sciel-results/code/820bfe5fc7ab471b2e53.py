import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_left (2x2) * B_top (2x10)
    # Split B_top into 5 blocks of 2x2
    # Strassen for 2x2x2
    # M1..M7
    
    # Coefficients for A (i in 0..1, j in 0..1)
    # Coefficients for B (j in 0..1, k in 0..1 relative to block)
    # Coefficients for C (i in 0..1, k in 0..1 relative to block)
    
    # Strassen terms:
    # M1: A(a+d), B(e+h)
    # M2: A(c+d), B(e)
    # M3: A(a), B(f-h)
    # M4: A(d), B(g-e)
    # M5: A(a+b), B(h)
    # M6: A(c-a), B(e+f)
    # M7: A(b-d), B(g+h)
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in 2x2 block
    # Global A index: i*3 + j
    
    # B indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global B index: j*10 + k
    
    # C indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global C index: k*2 + i
    
    strassen_A = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    strassen_B = [
        [1, 1, 0, 0], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    strassen_C = [
        # M1 M2 M3 M4 M5 M6 M7
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    for s in range(5):
        k_base = 2 * s
        for r in range(7):
            # A part
            for i in range(2):
                for j in range(2):
                    val = strassen_A[r][i*2 + j]
                    if val != 0:
                        U[i*3 + j][col] += val
                        
            # B part
            for j in range(2):
                for k_rel in range(2):
                    k = k_base + k_rel
                    val = strassen_B[r][j*2 + k_rel]
                    if val != 0:
                        V[j*10 + k][col] += val
                        
            # C part
            for i in range(2):
                for k_rel in range(2):
                    k = k_base + k_rel
                    val = strassen_C[i*2 + k_rel][r]
                    if val != 0:
                        W[k*2 + i][col] += val
                        
            col += 1
            
    # Part 2: A_right (2x1) * B_bot (1x10)
    # A_right is column 2 of A.
    # B_bot is row 2 of B.
    # C = A_right * B_bot
    # C[i, k] = A[i, 2] * B[2, k]
    
    for i in range(2):
        for k in range(10):
            U[i*3 + 2][col] = 1
            V[2*10 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
