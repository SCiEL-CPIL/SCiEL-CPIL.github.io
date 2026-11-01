import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds safety margin
    
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define Strassen terms for a 2x2 block
    # Each term is a tuple: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i, j): coeff}
    # V_coeffs: dict {(j, k_local): coeff}
    # W_coeffs: dict {(i, k_local): coeff}
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        ({(0,0): 1}, 
         {(0,1): 1, (1,1): -1}, 
         {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
        ({(1,1): 1}, 
         {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1}),
    ]
    
    # Iterate over the 4 blocks of B (columns 0-1, 2-3, 4-5, 6-7)
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        
        for r, (u_coeff, v_coeff, w_coeff) in enumerate(strassen_terms):
            t = q * 7 + r
            
            # Fill U
            for (i, j), coeff in u_coeff.items():
                U[i * m + j, t] = coeff
            
            # Fill V
            # j is row index of B (0 or 1)
            # k_local is 0 or 1, mapping to global k1 or k2
            for (j, k_local), coeff in v_coeff.items():
                k_global = k1 + k_local
                V[j * p + k_global, t] = coeff
            
            # Fill W
            # i is col index of C (0 or 1)
            # k_local is 0 or 1, mapping to global k1 or k2
            # W index is k_global * n + i
            for (i, k_local), coeff in w_coeff.items():
                k_global = k1 + k_local
                W[k_global * n + i, t] = coeff
    
    # Check if time is running out
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
