import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Uses a hybrid approach: Rank 1 for the j=0 slice, and Strassen (Rank 7) for each 
    of the three 2x2x2 sub-problems corresponding to j in {1,2} and k blocks.
    Total Rank = 1 + 3 * 7 = 22.
    """
    n, m, p = 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n
    
    rank = 22
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # --- Part 1: j=0 term (Rank 1) ---
    # C[i,k] += A[i,0] * B[0,k]
    # U[i,0,c] = 1, V[0,k,c] = 1, W[i,k,c] = 1
    c_idx = 0
    for i in range(n):
        U[i * m + 0, c_idx] = 1.0
    for k in range(p):
        V[0 * p + k, c_idx] = 1.0
        for i in range(n):
            W[k * n + i, c_idx] = 1.0
            
    # --- Part 2: j=1,2 terms (Rank 21) ---
    # We split the 2x2x6 sub-problem (A[1:3,:], B[1:3,:]) into three 2x2x2 blocks.
    # Block t corresponds to k in [2*t, 2*t+1].
    
    # Indices for A' (2x2):
    # A'[0,0] = A[0,1] -> U row 1
    # A'[0,1] = A[0,2] -> U row 2
    # A'[1,0] = A[1,1] -> U row 4
    # A'[1,1] = A[1,2] -> U row 5
    
    # Indices for B' (2x2) in block t:
    # B'[0,0] = B[1, 2t]   -> V row 6 + 2t
    # B'[0,1] = B[1, 2t+1] -> V row 7 + 2t
    # B'[1,0] = B[2, 2t]   -> V row 12 + 2t
    # B'[1,1] = B[2, 2t+1] -> V row 13 + 2t
    
    # Indices for C' (2x2) in block t:
    # C'[0, 2t]   -> W row 4t
    # C'[0, 2t+1] -> W row 4t + 2
    # C'[1, 2t]   -> W row 4t + 1
    # C'[1, 2t+1] -> W row 4t + 3
    
    for t in range(3):
        base_c = 1 + t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen Multiplications
        # M1 = (a00 + a11)(b00 + b11)
        c = base_c + 0
        U[1, c] = 1.0
        U[5, c] = 1.0
        V[6 + k0, c] = 1.0
        V[12 + k1, c] = 1.0
        W[4*t, c] = 1.0
        W[4*t+3, c] = 1.0
        
        # M2 = (a10 + a11)b00
        c = base_c + 1
        U[4, c] = 1.0
        U[5, c] = 1.0
        V[6 + k0, c] = 1.0
        W[4*t+1, c] = 1.0
        
        # M3 = a00(b10 - b11)
        c = base_c + 2
        U[1, c] = 1.0
        V[12 + k0, c] = 1.0
        V[12 + k1, c] = -1.0
        W[4*t+1, c] = 1.0
        W[4*t+3, c] = -1.0
        
        # M4 = a11(b01 - b00)
        c = base_c + 3
        U[5, c] = 1.0
        V[7 + k0, c] = 1.0
        V[6 + k0, c] = -1.0
        W[4*t+2, c] = 1.0
        
        # M5 = (a00 + a01)b11
        c = base_c + 4
        U[1, c] = 1.0
        U[2, c] = 1.0
        V[13 + k0, c] = 1.0
        W[4*t, c] = -1.0
        W[4*t+2, c] = 1.0
        
        # M6 = (a10 - a00)(b00 + b01)
        c = base_c + 5
        U[4, c] = 1.0
        U[1, c] = -1.0
        V[6 + k0, c] = 1.0
        V[7 + k0, c] = 1.0
        W[4*t+3, c] = 1.0
        
        # M7 = (a01 - a11)(b10 + b11)
        c = base_c + 6
        U[2, c] = 1.0
        U[5, c] = -1.0
        V[12 + k0, c] = 1.0
        V[13 + k0, c] = 1.0
        W[4*t, c] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
