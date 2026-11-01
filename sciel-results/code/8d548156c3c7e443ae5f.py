import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Theoretical minimum rank for M_{2,2,8} is 28 (4 * Strassen(2,2,2)).
    # We construct this deterministic decomposition.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen products for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: A[i,j], B[j,k], C[i,k]
    # Note: In our global tensor, B is 2x8, C is 2x8.
    # We partition B and C into 4 blocks of 2 columns.
    # For block q, B_sub is B[:, 2q:2q+2], C_sub is C[:, 2q:2q+2].
    
    strassen_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Check deadline before expensive verification
    if time.time() > deadline - 10:
        # Return without verification (high confidence in correctness)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the decomposition
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    if not np.allclose(T_computed, T_target, atol=1e-10):
        # Fallback to schoolbook (Rank 32) if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # Convert to lists for the return interface
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
