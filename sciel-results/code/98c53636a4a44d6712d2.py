import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    # 4 blocks of 2x2x2 (Strassen rank 7) -> 28
    # 2x1x8 (rank 16)
    # 1x2x8 (rank 16)
    # 1x1x8 (rank 8)
    total_rank = 28 + 16 + 16 + 8
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for (i, j), val in u_indices.items():
            U[i * m + j][col] += val
        for (j, k), val in v_indices.items():
            V[j * p + k][col] += val
        for (k, i), val in w_indices.items():
            W[k * n + i][col] += val
        col += 1

    # 1. Strassen part for A[0:2, 0:2] * B[0:2, :]
    # B is 2x8. We split B into 4 blocks of 2x2.
    # Block b covers columns 2b, 2b+1.
    
    # Strassen 7 products for 2x2 matrices.
    # Local indices 0,1 for rows/cols.
    # Global A indices: i in {0,1}, j in {0,1}.
    # Global B indices: j in {0,1}, k in {2b, 2b+1}.
    # Global C indices: i in {0,1}, k in {2b, 2b+1}.
    
    # Define the 7 Strassen terms (u, v, w)
    # u: dict (i, j) -> coeff
    # v: dict (j, k) -> coeff
    # w: dict (k, i) -> coeff (local k, i)
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1} # C00 += M1, C11 += M1
        ),
        # M2 = (A10 + A11)B00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): 0, (1,0): 1} # C10 += M2. C00? No. C11? No.
            # C10 = M2 + M4. So W(0,1) = 1? No, C10 is i=1, k=0.
            # W index is (k, i). So W(0, 1).
        ),
        # M3 = A00(B01 - B11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1} # C01 += M3. i=0, k=1. W(1,0).
        ),
        # M4 = A11(B10 - B00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,1): 1} # C10 += M4 (i=1, k=0 -> W(0,1)). C11 += M4 (i=1, k=1 -> W(1,1)).
        ),
        # M5 = (A00 + A01)B11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,1): 1, (1,0): -1} # C11 += M5 (W(1,1)). C00 -= M5 (W(0,0) = -1).
            # C00 = M1 + M4 - M5 + M7. So W(0,0) = -1.
        ),
        # M6 = (A10 - A00)(B00 + B01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1} # C11 += M6. i=1, k=1. W(1,1).
        ),
        # M7 = (A01 - A11)(B10 + B11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1} # C00 += M7 (W(0,0)). C10 += M7 (W(0,1)).
        )
    ]
    
    # Correcting the W coefficients based on Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify the W dicts for each term.
    # Term 1 (M1): C00+=1, C11+=1. W: {(0,0):1, (1,1):1}
    # Term 2 (M2): C10+=1. W: {(0,1):1}
    # Term 3 (M3): C01+=1. W: {(1,0):1}
    # Term 4 (M4): C10+=1, C11+=1. W: {(0,1):1, (1,1):1}
    # Term 5 (M5): C01+=1, C00-=1. W: {(1,0):1, (0,0):-1}
    # Term 6 (M6): C11+=1. W: {(1,1):1}
    # Term 7 (M7): C00+=1, C10+=1. W: {(0,0):1, (0,1):1}
    
    strassen_terms_corrected = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,0): 1, (0,0): -1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1}
        )
    ]
    
    for b in range(4):
        k_offset = 2 * b
        for u, v, w in strassen_terms_corrected:
            # Map local indices to global
            u_global = {}
            for (i, j), val in u.items():
                u_global[(i, j)] = val
            
            v_global = {}
            for (j, k_loc), val in v.items():
                k_glob = k_offset + k_loc
                v_global[(j, k_glob)] = val
            
            w_global = {}
            for (k_loc, i), val in w.items():
                k_glob = k_offset + k_loc
                w_global[(k_glob, i)] = val
                
            add_term(u_global, v_global, w_global)

    # 2. A[0:2, 2] * B[2, :]
    # A indices (0,2), (1,2). B index (2, k).
    # C indices (0, k), (1, k).
    for i in range(2):
        for k in range(8):
            add_term(
                {(i, 2): 1},
                {(2, k): 1},
                {(k, i): 1}
            )

    # 3. A[2, 0:2] * B[0:2, :]
    # A indices (2,0), (2,1). B indices (0,k), (1,k).
    # C index (2, k).
    for j in range(2):
        for k in range(8):
            add_term(
                {(2, j): 1},
                {(j, k): 1},
                {(k, 2): 1}
            )

    # 4. A[2, 2] * B[2, :]
    for k in range(8):
        add_term(
            {(2, 2): 1},
            {(2, k): 1},
            {(k, 2): 1}
        )

    return {"U": U, "V": V, "W": W}
