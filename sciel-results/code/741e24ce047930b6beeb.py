import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 6
    
    # Total rank
    rank = 26
    
    # Initialize U, V, W
    # U: (n*m, rank) = (6, 26)
    # V: (m*p, rank) = (18, 26)
    # W: (p*n, rank) = (12, 26)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to get indices
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i
        
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms as lists of (i, j, coeff) for U, (j, k, coeff) for V, (k, i, coeff) for W
    # Note: For Strassen, the "k" in the 2x2x2 block is local.
    
    strassen_terms = [
        # M1
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)] # C00, C01, C10, C11 coeffs: 1, 0, 0, 1? No.
            # Let's define W coeffs per output (k, i)
            # C00 (k=0, i=0): 1
            # C01 (k=1, i=0): 0
            # C10 (k=0, i=1): 0
            # C11 (k=1, i=1): 1
            # So W: (0,0,1), (1,1,1)
        },
        # M2
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1), (1,0,0)], # b00 is (0,0). b10 is (1,0).
            # M2 = (a10+a11)b00. So V is just b00.
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)] # C10 (k=0, i=1): 1. C11 (k=1, i=1): -1.
        },
        # M3
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)], # b01 - b11. b01 is (0,1), b11 is (1,1).
            'W': [(1,0,1), (1,1,1)] # C01 (k=1, i=0): 1. C11 (k=1, i=1): 1.
        },
        # M4
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)], # b10 - b00.
            'W': [(0,0,1), (0,1,1)] # C00 (k=0, i=0): 1. C10 (k=0, i=1): 1.
        },
        # M5
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)], # b11.
            'W': [(0,0,-1), (1,0,1)] # C00: -1. C01: 1.
        },
        # M6
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)], # b00 + b01.
            'W': [(1,1,1)] # C11: 1.
        },
        # M7
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)], # b10 + b11.
            'W': [(0,0,1), (1,1,1)] # C00: 1. C11: 1.
        }
    ]
    
    # Apply Strassen terms to Block 1 (k=0,1)
    for term in strassen_terms:
        # U
        for i, j, coeff in term['U']:
            U[u_idx(i, j), col] += coeff
        # V
        for j, k, coeff in term['V']:
            V[v_idx(j, k), col] += coeff
        # W
        for k, i, coeff in term['W']:
            W[w_idx(k, i), col] += coeff
        col += 1
        
    # Apply Outer Product terms to Block 1 (j=2, k=0,1,2)
    for i in range(2):
        for k in range(3):
            U[u_idx(i, 2), col] = 1
            V[v_idx(2, k), col] = 1
            W[w_idx(k, i), col] = 1
            col += 1
            
    # Apply Strassen terms to Block 2 (k=3,4)
    # Local k 0->3, 1->4
    for term in strassen_terms:
        # U is same
        for i, j, coeff in term['U']:
            U[u_idx(i, j), col] += coeff
        # V: k shifted by 3
        for j, k_loc, coeff in term['V']:
            k_glob = k_loc + 3
            V[v_idx(j, k_glob), col] += coeff
        # W: k shifted by 3
        for k_loc, i, coeff in term['W']:
            k_glob = k_loc + 3
            W[w_idx(k_glob, i), col] += coeff
        col += 1
        
    # Apply Outer Product terms to Block 2 (j=2, k=3,4,5)
    for i in range(2):
        for k in range(3, 6):
            U[u_idx(i, 2), col] = 1
            V[v_idx(2, k), col] = 1
            W[w_idx(k, i), col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
