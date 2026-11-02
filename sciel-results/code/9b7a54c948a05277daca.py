import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (p00+p11)(q00+q11)
    # M2 = (p10+p11)q00
    # M3 = p00(q01-q11)
    # M4 = p11(q10-q00)
    # M5 = (p00+p01)q11
    # M6 = (p10-p00)(q00+q01)
    # M7 = (p01-p11)(q10+q11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 products
    # Each product is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (r,s) to coeff for P[r,s]
    # v_coeffs: dict mapping (r,s) to coeff for Q[r,s]
    # w_coeffs: dict mapping (r,s) to coeff for C[r,s]
    
    products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}) # M7
    ]
    
    # Define the 8 sub-problems
    # (i0, j0, k0)
    # B_L: k0 in {0, 2}
    # B_R: k0 in {4, 6}
    # For each k0, we have two j0 values: 0 and 2
    # For each j0, we have two i0 values? No, i0 is always 0.
    # Wait, A is 2x4. i is 0,1.
    # The sub-problem is 2x2x2.
    # i0 is the row offset in A and C. Always 0.
    # j0 is the col offset in A and row offset in B.
    # k0 is the col offset in B and C.
    
    sub_problems = []
    for k_block in [0, 4]:
        for j_block in [0, 2]:
            # A_L (j=0,1) or A_R (j=2,3)
            # B_TL/TR (j=0,1) or B_BL/BR (j=2,3)
            # We need to pair A_L with B_T* and A_R with B_B*
            
            # Case 1: A_L (j0=0) with B_T* (j0=0)
            # Output k0
            sub_problems.append((0, 0, k_block))
            
            # Case 2: A_R (j0=2) with B_B* (j0=2)
            # Output k0
            sub_problems.append((0, 2, k_block))
            
            # Wait, this only covers C_L,L and C_R,L?
            # C_L = [C_LL, C_LR]
            # C_LL = A_L B_TL + A_R B_BL
            # C_LR = A_L B_TR + A_R B_BR
            
            # My loop above:
            # k_block=0 (B_L left half, cols 0-1)
            # j_block=0 -> A_L (cols 0-1) with B_TL (rows 0-1, cols 0-1). Correct for C_LL.
            # j_block=2 -> A_R (cols 2-3) with B_BL (rows 2-3, cols 0-1). Correct for C_LL.
            
            # k_block=2 (B_L right half, cols 2-3)
            # j_block=0 -> A_L (cols 0-1) with B_TR (rows 0-1, cols 2-3). Correct for C_LR.
            # j_block=2 -> A_R (cols 2-3) with B_BR (rows 2-3, cols 2-3). Correct for C_LR.
            
            # k_block=4 (B_R left half, cols 4-5)
            # j_block=0 -> A_L with B_TL' (rows 0-1, cols 4-5). Correct for C_RL.
            # j_block=2 -> A_R with B_BL' (rows 2-3, cols 4-5). Correct for C_RL.
            
            # k_block=6 (B_R right half, cols 6-7)
            # j_block=0 -> A_L with B_TR' (rows 0-1, cols 6-7). Correct for C_RR.
            # j_block=2 -> A_R with B_BR' (rows 2-3, cols 6-7). Correct for C_RR.
            
    # The list sub_problems has 8 entries.
    # For each, we add 7 rank-1 terms.
    
    for i0, j0, k0 in sub_problems:
        for u_c, v_c, w_c in products:
            # Add to U
            for (r, s), val in u_c.items():
                u_idx = (i0 + r) * m + (j0 + s)
                U[u_idx][col] += val
            
            # Add to V
            for (r, s), val in v_c.items():
                v_idx = (j0 + r) * p + (k0 + s)
                V[v_idx][col] += val
            
            # Add to W
            for (r, s), val in w_c.items():
                w_idx = (k0 + s) * n + (i0 + r)
                W[w_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
