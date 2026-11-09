"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Part 1: 4x4 * 4x4 using recursive Strassen -> rank 49
    # Part 2: 4x4 * 4x3 using schoolbook -> rank 48
    # Total rank: 97
    
    rank_total = 49 + 48
    
    U = [[0.0] * rank_total for _ in range(n * m)]  # 16 x 97
    V = [[0.0] * rank_total for _ in range(m * p)]  # 28 x 97
    W = [[0.0] * rank_total for _ in range(p * n)]  # 28 x 97
    
    # Rank-7 Strassen for 2x2*2x2->2x2
    # U2[i*2+j], V2[j*2+k], W2[k*2+i]
    U2 = [
        [1,  0,  1,  0,  1, -1,  0],  # A[0,0]
        [0,  0,  0,  0,  1,  0,  1],   # A[0,1]
        [0,  1,  0,  0,  0,  1,  0],   # A[1,0]
        [1,  1,  0,  1,  0,  0, -1],   # A[1,1]
    ]
    V2 = [
        [1,  1, -1,  0,  0,  1,  0],   # B[0,0]
        [1,  0,  0, -1,  0,  1,  0],   # B[0,1]
        [1,  0,  1,  0,  0,  0,  1],   # B[1,0]
        [1,  0,  0,  1,  1,  0,  1],   # B[1,1]
    ]
    W2 = [
        [1,  0,  0, -1,  0,  1, -1],   # C[0,0] (k=0,i=0)
        [0,  0,  1,  0,  1,  0,  0],   # C[0,1] (k=1,i=0)
        [0,  1,  0,  1,  0,  0,  0],   # C[1,0] (k=0,i=1)
        [1,  1,  0,  0, -1, 0,  0],   # C[1,1] (k=1,i=1)
    ]
    
    # Verify 2x2 Strassen: T[i,j,k] = sum_r U2[i*2+j][r]*V2[j*2+k][r]*W2[k*2+i][r] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j][r]*V2[j*2+k][r]*W2[k*2+i][r] for r in range(7))
                if abs(s - 1) > 1e-9:
                    # Fallback to schoolbook for 4x4x4 if Strassen is broken
                    rank_total = n * m * p
                    U = [[0.0] * rank_total for _ in range(n * m)]
                    V = [[0.0] * rank_total for _ in range(m * p)]
                    W = [[0.0] * rank_total for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    # Super-block Strassen definitions for 4x4x4
    M_defs = [
        # M1: X = A00+A11, Y = B00+B11
        (
            {(0,0): [(0,0,1),(1,1,1)], (0,1): [(0,0,1),(1,1,1)],
             (1,0): [(0,0,1),(1,1,1)], (1,1): [(0,0,1),(1,1,1)]},
            {(0,0): [(0,0,1),(1,1,1)], (0,1): [(0,0,1),(1,1,1)],
             (1,0): [(0,0,1),(1,1,1)], (1,1): [(0,0,1),(1,1,1)]}
        ),
        # M2: X = A10+A11, Y = B00
        (
            {(0,0): [(1,0,1),(1,1,1)], (0,1): [(1,0,1),(1,1,1)],
             (1,0): [(1,0,1),(1,1,1)], (1,1): [(1,0,1),(1,1,1)]},
            {(0,0): [(0,0,1)], (0,1): [(0,0,1)],
             (1,0): [(0,0,1)], (1,1): [(0,0,1)]}
        ),
        # M3: X = A00, Y = B01-B11
        (
            {(0,0): [(0,0,1)], (0,1): [(0,0,1)],
             (1,0): [(0,0,1)], (1,1): [(0,0,1)]},
            {(0,0): [(0,1,1),(1,1,-1)], (0,1): [(0,1,1),(1,1,-1)],
             (1,0): [(0,1,1),(1,1,-1)], (1,1): [(0,1,1),(1,1,-1)]}
        ),
        # M4: X = A11, Y = B10-B00
        (
            {(0,0): [(1,1,1)], (0,1): [(1,1,1)],
             (1,0): [(1,1,1)], (1,1): [(1,1,1)]},
            {(0,0): [(1,0,1),(0,0,-1)], (0,1): [(1,0,1),(0,0,-1)],
             (1,0): [(1,0,1),(0,0,-1)], (1,1): [(1,0,1),(0,0,-1)]}
        ),
        # M5: X = A00+A01, Y = B11
        (
            {(0,0): [(0,0,1),(0,1,1)], (0,1): [(0,0,1),(0,1,1)],
             (1,0): [(0,0,1),(0,1,1)], (1,1): [(0,0,1),(0,1,1)]},
            {(0,0): [(1,1,1)], (0,1): [(1,1,1)],
             (1,0): [(1,1,1)], (1,1): [(1,1,1)]}
        ),
        # M6: X = A10-A00, Y = B00+B01
        (
            {(0,0): [(1,0,1),(0,0,-1)], (0,1): [(1,0,1),(0,0,-1)],
             (1,0): [(1,0,1),(0,0,-1)], (1,1): [(1,0,1),(0,0,-1)]},
            {(0,0): [(0,0,1),(0,1,1)], (0,1): [(0,0,1),(0,1,1)],
             (1,0): [(0,0,1),(0,1,1)], (1,1): [(0,0,1),(0,1,1)]}
        ),
        # M7: X = A01-A11, Y = B10+B11
        (
            {(0,0): [(0,1,1),(1,1,-1)], (0,1): [(0,1,1),(1,1,-1)],
             (1,0): [(0,1,1),(1,1,-1)], (1,1): [(0,1,1),(1,1,-1)]},
            {(0,0): [(1,0,1),(1,1,1)], (0,1): [(1,0,1),(1,1,1)],
             (1,0): [(1,0,1),(1,1,1)], (1,1): [(1,0,1),(1,1,1)]}
        ),
    ]
    
    W_super = {
        (0, 0, 0): 1, (0, 1, 1): 1,
        (1, 1, 0): 1, (1, 1, 1): 1,
        (2, 0, 1): 1,
        (3, 0, 0): -1, (3, 1, 0): 1,
        (4, 0, 1): 1, (4, 1, 1): -1,
        (5, 0, 0): 1,
        (6, 0, 0): -1,
    }
    
    col_offset = 0
    for mi in range(7):
        X_def, Y_def = M_defs[mi]
        for ri in range(7):
            for a in range(2):
                for b in range(2):
                    for (br, bc, cx) in X_def[(a,b)]:
                        gi = 2*br + a
                        gj = 2*bc + b
                        U[gi*m + gj][col_offset] += cx * U2[a*2+b][ri]
            for c in range(2):
                for d in range(2):
                    for (br, bc, cy) in Y_def[(c,d)]:
                        gj = 2*br + c
                        gk = 2*bc + d
                        V[gj*p + gk][col_offset] += cy * V2[c*2+d][ri]
            for cb_r in range(2):
                for cb_c in range(2):
                    key = (mi, cb_r, cb_c)
                    if key in W_super:
                        wc = W_super[key]
                        for a in range(2):
                            for b in range(2):
                                gi = 2*cb_r + a
                                gk = 2*cb_c + b
                                W[gk*n + gi][col_offset] += wc * W2[b*2+a][ri]
            col_offset += 1
    
    # Part 2: Schoolbook for columns 4,5,6
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m + j][col_offset] = 1
                V[j*p + k][col_offset] = 1
                W[k*n + i][col_offset] = 1
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
