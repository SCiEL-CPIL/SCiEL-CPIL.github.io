import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for a 2x3x2 block
    # Term structure: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i, j, val)
    # v_coeffs: list of (j, k_local, val)
    # w_coeffs: list of (i, k_local, val)
    
    terms = []
    
    # 1. M1 = (a00 + a11)(b00 + b11) -> c00, c11
    terms.append((
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 1, 1)]
    ))
    
    # 2. M2 = (a10 + a11)b00 -> c10, c11
    terms.append((
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ))
    
    # 3. M3 = a00(b10 - b11) -> c01, c11
    terms.append((
        [(0, 0, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)]
    ))
    
    # 4. M4 = a11(b01 - b00) -> c00, c10
    terms.append((
        [(1, 1, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(0, 0, 1), (1, 0, 1)]
    ))
    
    # 5. M5 = (a00 + a01)b11 -> c00(-1), c01(1)
    terms.append((
        [(0, 0, 1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, -1), (0, 1, 1)]
    ))
    
    # 6. M6 = (a10 - a00)(b00 + b01) -> c11
    terms.append((
        [(1, 0, 1), (0, 0, -1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 1, 1)]
    ))
    
    # 7. M7 = (a01 - a11)(b10 + b11) -> c00(1), c11(-1)
    terms.append((
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 1, -1)]
    ))
    
    # 8. N1 = a02 b20 -> c00
    terms.append((
        [(0, 2, 1)],
        [(2, 0, 1)],
        [(0, 0, 1)]
    ))
    
    # 9. N2 = a02 b21 -> c01
    terms.append((
        [(0, 2, 1)],
        [(2, 1, 1)],
        [(0, 1, 1)]
    ))
    
    # 10. N3 = a12 b20 -> c10
    terms.append((
        [(1, 2, 1)],
        [(2, 0, 1)],
        [(1, 0, 1)]
    ))
    
    # 11. N4 = a12 b21 -> c11
    terms.append((
        [(1, 2, 1)],
        [(2, 1, 1)],
        [(1, 1, 1)]
    ))
    
    # Fill the matrices for each of the 3 blocks
    for q in range(3):
        for r_local in range(11):
            r_global = q * 11 + r_local
            u_coeffs, v_coeffs, w_coeffs = terms[r_local]
            
            for i, j, val in u_coeffs:
                U[i * m + j][r_global] += val
                
            for j, k_local, val in v_coeffs:
                k_global = q * 2 + k_local
                V[j * p + k_global][r_global] += val
                
            for i, k_local, val in w_coeffs:
                k_global = q * 2 + k_local
                W[k_global * n + i][r_global] += val
                
    return {"U": U, "V": V, "W": W}
