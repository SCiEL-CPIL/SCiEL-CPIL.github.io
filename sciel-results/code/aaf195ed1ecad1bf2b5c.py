import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 is the sum of three rank-7 Strassen decompositions
    # for the three 2x2 blocks of B.
    # Since the V and W indices for each block are disjoint,
    # the ranks add up: 7 + 7 + 7 = 21.
    
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W local indices: (i, k_local)
    
    strassen = [
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # M3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # M4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            # Fill U
            for idx, val in u_coeffs.items():
                U[idx][r] = val
                
            # Fill V
            # Local B index l maps to global V index:
            # l=0 (B00) -> j=0, k=2b -> 0*p + 2b
            # l=1 (B01) -> j=0, k=2b+1 -> 0*p + 2b+1
            # l=2 (B10) -> j=1, k=2b -> 1*p + 2b
            # l=3 (B11) -> j=1, k=2b+1 -> 1*p + 2b+1
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 2 * b
                elif l == 1:
                    v_idx = 2 * b + 1
                elif l == 2:
                    v_idx = p + 2 * b
                elif l == 3:
                    v_idx = p + 2 * b + 1
                V[v_idx][r] = val
                
            # Fill W
            # W index is k*n + i
            # k_global = 2b + k_local
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
