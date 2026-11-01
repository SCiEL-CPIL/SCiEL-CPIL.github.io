"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 Strassen decomposition for M_{2,2,6}.
    Rank 20 is not achievable for this tensor.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Build the rank-21 Strassen decomposition
    rank21 = _build_rank21_strassen(n, m, p)
    
    return rank21

def _build_rank21_strassen(n, m, p):
    """Build the rank-21 Strassen decomposition."""
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
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
            
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}

def _to_rational(matrix, tol=1e-6):
    """Try to convert a matrix to integer/half-integer values."""
    if matrix is None:
        return None
    
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            # Try to round to nearest half-integer
            rounded = round(val * 2) / 2
            if abs(val - rounded) < tol:
                new_row.append(rounded)
            else:
                return None  # Not close to half-integer
        result.append(new_row)
    return result
# EVOLVE-BLOCK-END
