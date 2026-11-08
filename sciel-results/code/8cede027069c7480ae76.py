import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Hardcoded explicit rank-1 tensors for speed and exactness.
    """
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 4, 4
    N_A = n * m  # 16
    N_B = m * p  # 16
    N_C = p * n  # 16
    RANK = 49
    
    U = np.zeros((N_A, RANK))
    V = np.zeros((N_B, RANK))
    W = np.zeros((N_C, RANK))
    
    # Define the 7 Strassen terms for 2x2 matrices
    # Each term is (A_list, B_list, C_list)
    # Lists contain (row, col, coeff)
    # Note: For C, we store (row, col) of the matrix C, and convert to index later.
    
    strassen_terms = [
        # M1: (A00 + A11)(B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10 + A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3: A00(B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4: A11(B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5: (A00 + A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10 - A00)(B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01 - A11)(B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # The 49 rank-1 terms are combinations of the 7 block-level terms and 7 scalar-level terms.
    # Block level term i, Scalar level term j.
    # Total rank index k = i * 7 + j.
    
    for i in range(7):
        A_blocks, B_blocks, C_blocks = strassen_terms[i]
        for j in range(7):
            A_scalars, B_scalars, C_scalars = strassen_terms[j]
            
            k = i * 7 + j
            
            # Populate U
            for (br, bc, ca) in A_blocks:
                for (sr, sc, cb) in A_scalars:
                    row = 2 * br + sr
                    col = 2 * bc + sc
                    idx = row * 4 + col
                    U[idx, k] += ca * cb
            
            # Populate V
            for (br, bc, cb) in B_blocks:
                for (sr, sc, cc) in B_scalars:
                    row = 2 * br + sr
                    col = 2 * bc + sc
                    idx = row * 4 + col
                    V[idx, k] += cb * cc
            
            # Populate W
            # Trace convention: W[col * n + row]
            for (br, bc, cc) in C_blocks:
                for (sr, sc, cd) in C_scalars:
                    row = 2 * br + sr
                    col = 2 * bc + sc
                    idx = col * 4 + row
                    W[idx, k] += cc * cd
    
    # Verification (optional, kept for safety but fast)
    # T = np.einsum('as,bs,cs->abc', U, V, W)
    # Check against known tensor? 
    # The construction is mathematically exact for Strassen.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
