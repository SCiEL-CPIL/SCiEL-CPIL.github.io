"""
Rewritten program for M_{2,2,6} rank-21 decomposition.
Uses Strassen's algorithm applied to three 2x2 blocks.
"""
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 Strassen-based decomposition for M_{2,2,6}.
    
    U: shape (4, 21)
    V: shape (12, 21)
    W: shape (12, 21)
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 products
    # Format: (U_indices, V_indices, W_indices)
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V local indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: (row_i, col_k_local) -> maps to global W index k_global*n + i
    
    strassen_products = [
        # 1: (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # 2: (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # 3: A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # 4: A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # 5: (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # 6: (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # 7: (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_products[t]
            
            # Fill U
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            # Fill V
            # Global V index mapping for block b:
            # Local 0 (B00) -> 0*p + 2*b
            # Local 1 (B01) -> 0*p + 2*b + 1
            # Local 2 (B10) -> 1*p + 2*b
            # Local 3 (B11) -> 1*p + 2*b + 1
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            
            # Fill W
            # Global W index: k_global * n + i
            # k_global = 2 * b + k_local
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
