import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 construction
    rank = 44
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1 B1 (j=0)
    # A1 is 2x1, B1 is 1x8
    # Rank 16
    col = 0
    for i in range(n):
        for k in range(p):
            # U index: i * m + 0
            U[i * m + 0][col] = 1
            # V index: 0 * p + k
            V[0 * p + k][col] = 1
            # W index: k * n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 B2 (j=1,2)
    # A2 is 2x2, B2 is 2x8
    # Split B2 into 4 blocks of 2x2
    # Each block uses Strassen (rank 7)
    # Total 4 * 7 = 28
    
    # Strassen coefficients for 2x2x2
    # s: 0..6
    # U_coeffs: list of (local_i, local_j, coeff)
    # V_coeffs: list of (local_j, local_k, coeff)
    # W_coeffs: list of (local_k, local_i, coeff)
    
    strassen_data = [
        # s=0
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # s=1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (0,1,1), (1,1,-1) ]), # z10=+1, z11=-1 -> W(0,1)=1, W(1,1)=-1? 
        # Wait, W index is k*n + i.
        # z10 is k'=0, i'=1 -> W index 0*2+1 = 1.
        # z11 is k'=1, i'=1 -> W index 1*2+1 = 3.
        # So W coeffs should be mapped to global W indices in the loop.
        # Let's store local k', i' and coeff.
        # I will re-verify the W mapping in the loop.
        # Local z_{i', k'} -> W[k' * n + i']
        
        # Let's redefine strassen_data with local indices (i', j') for U, (j', k') for V, (k', i') for W.
    ]
    
    # Correct Strassen Data Structure
    # U: (i', j')
    # V: (j', k')
    # W: (k', i')
    
    S = [
        # M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # M2
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),
        # M3
        ([ (0,0,1) ], [ (1,0,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ]),
        # M4
        ([ (1,1,1) ], [ (0,1,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]),
        # M5
        ([ (0,0,1), (1,0,1) ], [ (1,1,1) ], [ (0,0,-1), (1,0,1) ]),
        # M6
        ([ (1,1,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ], [ (1,1,1) ]),
        # M7
        ([ (0,1,1), (1,1,-1) ], [ (1,1,1), (0,1,1) ], [ (0,0,1) ])
    ]
    
    for b in range(4):
        # Block b corresponds to k in [2b, 2b+1]
        for s in range(7):
            col = 16 + b * 7 + s
            u_data, v_data, w_data = S[s]
            
            # U: local i', j' -> global i, j = 1 + j'
            for li, lj, coeff in u_data:
                gi = li
                gj = 1 + lj
                U[gi * m + gj][col] += coeff
                
            # V: local j', k' -> global j = 1 + j', k = 2b + k'
            for lj, lk, coeff in v_data:
                gj = 1 + lj
                gk = 2 * b + lk
                V[gj * p + gk][col] += coeff
                
            # W: local k', i' -> global k = 2b + k', i = i'
            for lk, li, coeff in w_data:
                gk = 2 * b + lk
                gi = li
                W[gk * n + gi][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
