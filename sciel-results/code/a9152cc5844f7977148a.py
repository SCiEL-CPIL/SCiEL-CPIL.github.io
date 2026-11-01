import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for A (u_q)
    # A indices: 0:a00, 1:a01, 2:a10, 3:a11
    U_cols = [
        [1, 0, 0, 1],  # q=0: a00 + a11
        [0, 0, 1, 1],  # q=1: a10 + a11
        [1, 0, 0, 0],  # q=2: a00
        [0, 0, 0, 1],  # q=3: a11
        [1, 1, 0, 0],  # q=4: a00 + a01
        [-1, 0, 1, 0], # q=5: a10 - a00
        [0, 1, 0, -1]  # q=6: a01 - a11
    ]
    
    # For each block r (0 to 3)
    for r in range(4):
        # V coefficients for B (v_{r,q})
        # B indices: j*p + k. j in {0,1}, k in {0..7}
        # Block r uses k=2r, 2r+1
        k0 = 2 * r
        k1 = 2 * r + 1
        
        # Indices in V
        # b_0,k0 -> 0*8 + k0 = k0
        # b_0,k1 -> 0*8 + k1 = k1
        # b_1,k0 -> 1*8 + k0 = 8+k0
        # b_1,k1 -> 1*8 + k1 = 8+k1
        
        V_cols = [
            {k0: 1, 8 + k1: 1},       # q=0: b_0,k0 + b_1,k1
            {k0: 1},                 # q=1: b_0,k0
            {8 + k0: 1, 8 + k1: -1},    # q=2: b_1,k0 - b_1,k1
            {k1: 1, k0: -1},        # q=3: b_0,k1 - b_0,k0
            {8 + k1: 1},               # q=4: b_1,k1
            {k0: 1, k1: 1},         # q=5: b_0,k0 + b_0,k1
            {8 + k0: 1, 8 + k1: 1}      # q=6: b_1,k0 + b_1,k1
        ]
        
        # W coefficients for C (w_{i,k})
        # C indices: k*n + i. n=2.
        # c_0,k0 -> k0*2 + 0 = 2*k0
        # c_1,k0 -> k0*2 + 1 = 2*k0 + 1
        # c_0,k1 -> k1*2 + 0 = 2*k1
        # c_1,k1 -> k1*2 + 1 = 2*k1 + 1
        
        # Formulas for block r:
        # c_0,k0 = P0 + P3 - P4 + P6
        # c_0,k1 = P2 + P4
        # c_1,k0 = P1 + P5
        # c_1,k1 = P0 - P1 + P2 + P5
        
        # Map q to contributions to c_0,k0, c_0,k1, c_1,k0, c_1,k1
        W_contribs = [
            {2 * k0: 1, 2 * k1 + 1: 1},    # q=0
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}, # q=1
            {2 * k1: 1, 2 * k1 + 1: 1},    # q=2
            {2 * k0: 1},               # q=3
            {2 * k0: -1, 2 * k1: 1},     # q=4
            {2 * k0 + 1: 1, 2 * k1 + 1: 1},  # q=5
            {2 * k0: 1}                # q=6
        ]
        
        for q in range(7):
            col_idx = 7 * r + q
            
            # Fill U
            for idx, val in enumerate(U_cols[q]):
                U[idx, col_idx] = val
                
            # Fill V
            for idx, val in V_cols[q].items():
                V[idx, col_idx] = val
                
            # Fill W
            for idx, val in W_contribs[q].items():
                W[idx, col_idx] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
