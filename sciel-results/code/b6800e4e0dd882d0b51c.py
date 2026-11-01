import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 14 (7 for each 2x2x2 block)
    R = 14
    
    # Initialize matrices with zeros
    # U: (n*m, R) = (4, 14)
    # V: (m*p, R) = (8, 14)
    # W: (p*n, R) = (8, 14)
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)

    # Strassen factors for 2x2x2 multiplication
    # Local indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B11-B00)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Coefficients for U (A factors)
    # col r corresponds to M_{r+1}
    U_factors = [
        [1, 0, 0, 1],  # M1: A00, A11
        [0, 0, 1, 1],  # M2: A10, A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00, A01
        [1, 0, -1, 0], # M6: A10, A00
        [0, 1, 0, -1]  # M7: A01, A11
    ]
    
    # Coefficients for V (B factors)
    V_factors = [
        [1, 0, 0, 1],  # M1: B00, B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B11, B00
        [0, 0, 1, -1], # M4: B10, B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00, B01
        [0, 0, 1, 1]   # M7: B10, B11
    ]
    
    # Coefficients for W (C factors)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_factors = [
        [1, 0, 0, 1],   # M1: C00, C11
        [0, 0, 1, -1],  # M2: C10, C11
        [0, 1, 0, 1],   # M3: C01, C11
        [1, 0, 1, 0],   # M4: C00, C10
        [-1, 1, 0, 0],  # M5: C00, C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]

    # Block 1: Cols 0,1 of B (global B indices 0,1,4,5)
    # Global A indices: 0,1,2,3
    # Global B indices: 0,1,4,5
    # Global W indices: 0,1,2,3
    
    for r in range(7):
        # U part
        for a_idx in range(4):
            if U_factors[r][a_idx] != 0:
                U[a_idx, r] = U_factors[r][a_idx]
        
        # V part
        # Local B indices 0,1,2,3 map to Global 0,1,4,5
        b_map = [0, 1, 4, 5]
        for b_idx in range(4):
            if V_factors[r][b_idx] != 0:
                V[b_map[b_idx], r] = V_factors[r][b_idx]
                
        # W part
        # Local C indices 0,1,2,3 map to Global 0,1,2,3
        w_map = [0, 1, 2, 3]
        for w_idx in range(4):
            if W_factors[r][w_idx] != 0:
                W[w_map[w_idx], r] = W_factors[r][w_idx]

    # Block 2: Cols 2,3 of B (global B indices 2,3,6,7)
    # Global A indices: 0,1,2,3 (same A)
    # Global B indices: 2,3,6,7
    # Global W indices: 4,5,6,7
    
    for r in range(7):
        r_offset = r + 7
        
        # U part (A is shared, but we add to the new rank columns)
        for a_idx in range(4):
            if U_factors[r][a_idx] != 0:
                U[a_idx, r_offset] = U_factors[r][a_idx]
                
        # V part
        # Local B indices 0,1,2,3 map to Global 2,3,6,7
        b_map = [2, 3, 6, 7]
        for b_idx in range(4):
            if V_factors[r][b_idx] != 0:
                V[b_map[b_idx], r_offset] = V_factors[r][b_idx]
                
        # W part
        # Local C indices 0,1,2,3 map to Global 4,5,6,7
        w_map = [4, 5, 6, 7]
        for w_idx in range(4):
            if W_factors[r][w_idx] != 0:
                W[w_map[w_idx], r_offset] = W_factors[r][w_idx]

    # Convert to lists of lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
