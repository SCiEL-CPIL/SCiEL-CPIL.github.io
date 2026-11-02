import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a hybrid decomposition.
    # 6 disjoint 2x2x2 blocks using Strassen (rank 7 each)
    # Remaining 78 terms using schoolbook (rank 1 each)
    # Total rank = 6 * 7 + 78 = 42 + 78 = 120? 
    # Wait.
    # Schoolbook for 2x2x2 is 8. Strassen is 7.
    # If we use Strassen for 6 blocks, we use 6*7=42 rank for 6*8=48 terms.
    # The remaining terms are 105 - 48 = 57 terms.
    # Schoolbook rank for these 57 is 57.
    # Total rank = 42 + 57 = 99.
    
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 6 blocks
    # i offsets: 0
    # j offsets: 0, 2
    # k offsets: 0, 2, 4
    
    blocks = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                blocks.append((i0, j0, k0))
                
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # U coeffs (i, j)
    # V coeffs (j, k)
    # W coeffs (k, i)
    
    # Local indices 0,1
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1) -> Index k*n+i
    
    strassen_terms = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        # M1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (0,1), (1,0), (1,1)], [1, 1, 1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(0,0), (1,0)], [1, 1]),
        # M3
        ([(0,0)], [1], [(1,0), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # M4
        ([(1,1)], [1], [(0,1), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1]),
        # M5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (1,0)], [-1, -1]),
        # M6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0), (0,1)], [1, -1]),
    ]
    
    # Note: W index is k*n + i.
    # Local (k, i) -> (0,0), (0,1), (1,0), (1,1)
    # Global index: (k0+k)*n + (i0+i)
    
    for i0, j0, k0 in blocks:
        for term in strassen_terms:
            u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff = term
            for idx, coeff in zip(u_idx, u_coeff):
                li, lj = idx
                U[(i0 + li) * m + (j0 + lj)][col] += coeff
            for idx, coeff in zip(v_idx, v_coeff):
                lj, lk = idx
                V[(j0 + lj) * p + (k0 + lk)][col] += coeff
            for idx, coeff in zip(w_idx, w_coeff):
                lk, li = idx
                W[(k0 + lk) * n + (i0 + li)][col] += coeff
            col += 1
            
    # Remaining terms: Schoolbook
    # Iterate all (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i, j, k) is in any block
                in_block = False
                for i0, j0, k0 in blocks:
                    if i0 <= i < i0 + 2 and j0 <= j < j0 + 2 and k0 <= k < k0 + 2:
                        in_block = True
                        break
                if not in_block:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
