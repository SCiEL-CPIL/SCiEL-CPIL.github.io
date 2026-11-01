"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for each 2x2 block of B.
    # B is 2x16, so we have 8 blocks of 2x2.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the linear forms for A (u) and B (v) and C (w)
    # Indices for U: i*m + j. i in {0,1}, j in {0,1}.
    # 0:00, 1:01, 2:10, 3:11
    # Indices for V: j*p + k. j in {0,1}, k in {0..15}.
    # For block t (0..7), k is 2t or 2t+1.
    # Indices for W: k*n + i. k in {0..15}, i in {0,1}.
    # For block t, k is 2t or 2t+1.
    
    # Strassen terms for one block:
    # term 0: u = [1, 0, 0, 1] (a+d), v = [1, 0, 0, 1] (e+h) -> w = [1, 0, 0, 1] (C11+C22 part)
    # Actually, let's map the output C11, C12, C21, C22 to W indices.
    # W index for C[i][k]. i is row, k is col.
    # In a block, local k is 0 or 1. Global k is 2t or 2t+1.
    # C[0][2t] -> W[2t*n + 0] = W[4t]
    # C[0][2t+1] -> W[2t*n + 1] = W[4t+1]
    # C[1][2t] -> W[2t*n + 2] = W[4t+2]
    # C[1][2t+1] -> W[2t*n + 3] = W[4t+3]
    
    # Let's define the 7 Strassen terms for a generic block.
    # u is in R^4 (A indices 00,01,10,11)
    # v is in R^4 (B local indices 00,01,10,11)
    # w is in R^4 (C local indices 00,01,10,11)
    
    strassen_terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])  # M7
    ]
    
    # Verify M4: d(g-e). u=[0,0,0,1], v=[0,-1,1,0]. w: C11=1, C21=1.
    # C11 = M1+M4-M5+M7. C21 = M2+M4.
    # M4 contributes to C11 and C21.
    # My w for M4 is [1, 0, 1, 0]. Correct.
    
    # Verify M5: (a+b)h. u=[1,1,0,0], v=[0,0,0,1].
    # C11 = ... - M5. C12 = M3+M5.
    # w for M5: C11=-1, C12=1. [ -1, 1, 0, 0 ]. Correct.
    
    # Verify M7: (b-d)(g+h). u=[0,1,0,-1], v=[0,0,1,1].
    # C11 = ... + M7. C22 = ... - M7.
    # w for M7: C11=1, C22=-1. [ 1, 0, 0, -1 ]. Correct.
    
    for t in range(8):
        for s in range(7):
            col = t * 7 + s
            u, v, w = strassen_terms[s]
            
            # Fill U
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V
            # Local B indices 0..3 correspond to global indices:
            # 0: j=0, k=2t -> 0*p + 2t = 2t
            # 1: j=0, k=2t+1 -> 2t+1
            # 2: j=1, k=2t -> 1*p + 2t = 16 + 2t
            # 3: j=1, k=2t+1 -> 16 + 2t + 1
            v_indices = [2*t, 2*t+1, 16+2*t, 16+2*t+1]
            for i in range(4):
                V[v_indices[i]][col] = v[i]
                
            # Fill W
            # Local C indices 0..3 correspond to global indices:
            # 0: i=0, k=2t -> 2t*n + 0 = 4t
            # 1: i=0, k=2t+1 -> 2t*n + 1 = 4t+1
            # 2: i=1, k=2t -> 2t*n + 2 = 4t+2
            # 3: i=1, k=2t+1 -> 2t*n + 3 = 4t+3
            w_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
            for i in range(4):
                W[w_indices[i]][col] = w[i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
