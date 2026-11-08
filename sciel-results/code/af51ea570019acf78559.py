import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2 blocks
    # A is 2x4, B is 4x8
    # A = [A0, A1] where A0, A1 are 2x2
    # B is 4x8, split into 4 blocks of 2x2 horizontally?
    # B rows 0,1 and cols 2i, 2i+1 is B_0i
    # B rows 2,3 and cols 2i, 2i+1 is B_1i
    # C_i = A0 B_0i + A1 B_1i
    # Each product is 2x2 * 2x2, rank 7 via Strassen.
    # Total rank 4 * 2 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms (u, v, w) for 2x2 multiplication
    # u, v, w are length 4 vectors corresponding to 00, 01, 10, 11
    terms = [
        ([1, 0, 0, -1], [0, 0, -1, 1], [1, 0, 0, 1]),  # M1
        ([1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, 1], [0, 0, 1, -1], [1, 0, 0, -1])   # M7
    ]
    
    col = 0
    for i in range(4):
        # Block i of C corresponds to columns 2i, 2i+1 of C
        # W indices for C_i (local 00, 01, 10, 11 -> global k*n+i)
        # C local (r, c) -> Global C[2i+c, r]? No.
        # C is 2x8. C[row, col].
        # Block i is C[:, 2i:2i+2].
        # Local C_00 is C[0, 2i]. Global W index: col=2i, row=0 -> 2i*2 + 0 = 4i
        # Local C_01 is C[0, 2i+1]. Global W index: col=2i+1, row=0 -> (2i+1)*2 + 0 = 4i+2
        # Local C_10 is C[1, 2i]. Global W index: col=2i, row=1 -> 2i*2 + 1 = 4i+1
        # Local C_11 is C[1, 2i+1]. Global W index: col=2i+1, row=1 -> (2i+1)*2 + 1 = 4i+3
        
        w_indices = [4*i, 4*i+2, 4*i+1, 4*i+3]
        
        # First product: A0 * B_0i
        # A0 is A[:, 0:2]
        # A local 00->A[0,0] (idx 0), 01->A[0,1] (idx 1), 10->A[1,0] (idx 4), 11->A[1,1] (idx 5)
        u_indices_A0 = [0, 1, 4, 5]
        
        # B_0i is B[0:2, 2i:2i+2]
        # B local 00->B[0, 2i] (idx 0*8+2i), 01->B[0, 2i+1] (idx 0*8+2i+1)
        # B local 10->B[1, 2i] (idx 1*8+2i), 11->B[1, 2i+1] (idx 1*8+2i+1)
        v_indices_B0i = [2*i, 2*i+1, 8+2*i, 9+2*i]
        
        for u, v, w in terms:
            for k in range(4):
                if u[k] != 0:
                    U[u_indices_A0[k]][col] += u[k]
                if v[k] != 0:
                    V[v_indices_B0i[k]][col] += v[k]
                if w[k] != 0:
                    W[w_indices[k]][col] += w[k]
            col += 1
            
        # Second product: A1 * B_1i
        # A1 is A[:, 2:4]
        # A local 00->A[0,2] (idx 2), 01->A[0,3] (idx 3), 10->A[1,2] (idx 6), 11->A[1,3] (idx 7)
        u_indices_A1 = [2, 3, 6, 7]
        
        # B_1i is B[2:4, 2i:2i+2]
        # B local 00->B[2, 2i] (idx 2*8+2i), 01->B[2, 2i+1] (idx 2*8+2i+1)
        # B local 10->B[3, 2i] (idx 3*8+2i), 11->B[3, 2i+1] (idx 3*8+2i+1)
        v_indices_B1i = [16+2*i, 17+2*i, 24+2*i, 25+2*i]
        
        for u, v, w in terms:
            for k in range(4):
                if u[k] != 0:
                    U[u_indices_A1[k]][col] += u[k]
                if v[k] != 0:
                    V[v_indices_B1i[k]][col] += v[k]
                if w[k] != 0:
                    W[w_indices[k]][col] += w[k]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
