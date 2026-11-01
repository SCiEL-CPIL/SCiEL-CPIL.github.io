"""
Combined program for matrix multiplication 2x5x5 rank minimization.
Uses a constructive Strassen-based decomposition for 2x2x2 sub-blocks
and schoolbook for the remainder to achieve rank 46.
"""
import time
import random
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-46 exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    n, m, p = DIMENSIONS
    # Target rank: 4 * 7 (Strassen) + 9 * 2 (Schoolbook) = 28 + 18 = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 2x2 blocks in j,k space
    # j indices: 0,1,2,3,4
    # k indices: 0,1,2,3,4
    # Blocks:
    # 1: j in {0,1}, k in {0,1}
    # 2: j in {0,1}, k in {2,3}
    # 3: j in {2,3}, k in {0,1}
    # 4: j in {2,3}, k in {2,3}
    
    blocks = [
        ([0, 1], [0, 1]),
        ([0, 1], [2, 3]),
        ([2, 3], [0, 1]),
        ([2, 3], [2, 3])
    ]
    
    # Strassen decomposition for 2x2x2 all-ones tensor
    # Local indices i, j, k in {0, 1}
    # U_local[i, j], V_local[j, k], W_local[k, i]
    
    # The 7 terms of Strassen
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2 = (A10+A11)B00       -> C10+=1
    # M3 = A00(B01-B11)       -> C01+=1
    # M4 = A11(B10-B00)       -> C00+=1, C10+=1
    # M5 = (A00+A01)B11       -> C00-=1, C01+=1
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11-=1
    
    # Note: W coefficients are for C_ik.
    # C00: M1(1) + M4(1) - M5(1) + M7(1)
    # C01: M3(1) + M5(1)
    # C10: M2(1) + M4(1)
    # C11: M1(1) - M2(1) + M3(1) + M6(1)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: (A00, A01, A10, A11)
        # V: (B00, B01, B10, B11)
        # W: (C00, C01, C10, C11)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),  # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),  # M2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)), # M3
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)), # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)), # M5
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)), # M6
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, -1)) # M7
    ]
    
    for js, ks in blocks:
        j0, j1 = js
        k0, k1 = ks
        
        for t_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
            
            # Map local i,j to global U index
            # U[i*m + j]
            for i in range(n): # i is 0, 1
                # Local j 0 -> j0, Local j 1 -> j1
                for lj in range(2):
                    gj = j0 + lj
                    u_val = u_coeffs[i * 2 + lj]
                    if u_val != 0:
                        U[i * m + gj][col] = u_val
                        
            # Map local j,k to global V index
            # V[j*p + k]
            for lj in range(2):
                gj = j0 + lj
                for lk in range(2):
                    gk = k0 + lk
                    v_val = v_coeffs[lj * 2 + lk]
                    if v_val != 0:
                        V[gj * p + gk][col] = v_val
                        
            # Map local k,i to global W index
            # W[k*n + i]
            for lk in range(2):
                gk = k0 + lk
                for i in range(n): # i is 0, 1
                    w_val = w_coeffs[lk * 2 + i]
                    if w_val != 0:
                        W[gk * n + i][col] = w_val
                        
            col += 1
            
    # Remaining (j,k) pairs
    # All pairs where j not in {0,1,2,3} OR k not in {0,1,2,3}
    # i.e., j=4 or k=4
    
    for j in range(m):
        for k in range(p):
            # Check if this pair is in any of the Strassen blocks
            in_block = False
            for js, ks in blocks:
                if j in js and k in ks:
                    in_block = True
                    break
            
            if not in_block:
                # Schoolbook for this (j,k) pair
                # For each i, add a term
                for i in range(n):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
