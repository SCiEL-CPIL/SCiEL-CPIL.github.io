"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 Strassen-based decomposition for M_{2,2,6}.
    
    The matrix multiplication tensor for (2x2) @ (2x6) is decomposed into
    three independent 2x2 block multiplications, each using Strassen's
    7-product algorithm.
    
    U: shape (n*m, rank) = (4, 21)
    V: shape (m*p, rank) = (12, 21)
    W: shape (p*n, rank) = (12, 21)
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 products
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B local indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: (row i, col k_local)
    
    strassen = [
        # Product 1: (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # Product 2: (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # Product 3: A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # Product 4: A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # Product 5: (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # Product 6: (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # Product 7: (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):  # Three 2x2 blocks
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            # Fill U entries
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            # Fill V entries
            # Local B index l maps to global V index:
            # B is 2x6, divided into three 2x2 blocks
            # l=0 (B00): row 0, col 2b
            # l=1 (B01): row 0, col 2b+1
            # l=2 (B10): row 1, col 2b
            # l=3 (B11): row 1, col 2b+1
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
            
            # Fill W entries
            # W index = k*n + i, where k is the output column and i is the output row
            # k_global = 2b + k_local
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
