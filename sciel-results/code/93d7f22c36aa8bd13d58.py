import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (8, rank)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 2] * B[2, :]
    # A is 2x3, B is 3x4.
    # A[:, 2] is 2x1. B[2, :] is 1x4.
    # Product is 2x4.
    # Terms: A[i, 2] * B[2, k] for i in 0..1, k in 0..3.
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            U[i * m + 2][col] = 1
            # V index: 2*p + k
            V[2 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 0:2] * B[0:2, :]
    # A_sub is 2x2 (cols 0,1). B_sub is 2x4 (rows 0,1).
    # Split B_sub into B_left (cols 0,1) and B_right (cols 2,3).
    # Two Strassen instances.
    
    def add_strassen(A_col_offset, B_row_offset, B_col_offset):
        nonlocal col
        # A_sub indices:
        # A[0, A_col_offset], A[0, A_col_offset+1]
        # A[1, A_col_offset], A[1, A_col_offset+1]
        
        # B_sub indices:
        # B[B_row_offset, B_col_offset], B[B_row_offset, B_col_offset+1]
        # B[B_row_offset+1, B_col_offset], B[B_row_offset+1, B_col_offset+1]
        
        # Strassen Multipliers:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Coefficients for C:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Helper to add a term to U, V, W
        def add_term(u_indices, v_indices, w_indices, coeff_w):
            nonlocal col
            # u_indices: list of (idx, coeff)
            # v_indices: list of (idx, coeff)
            # w_indices: list of (idx, coeff)
            
            # Note: U and V entries are sums of coefficients.
            # In tensor decomposition, U[i,c] and V[j,c] can be any real numbers.
            # Here we are constructing integer/half-integer solutions.
            
            for idx, c in u_indices:
                U[idx][col] += c
            for idx, c in v_indices:
                V[idx][col] += c
            for idx, c in w_indices:
                W[idx][col] += c
            col += 1

        # Indices in local 2x2 space:
        # A00 -> i=0, j=A_col_offset
        # A01 -> i=0, j=A_col_offset+1
        # A10 -> i=1, j=A_col_offset
        # A11 -> i=1, j=A_col_offset+1
        
        # B00 -> i=B_row_offset, k=B_col_offset
        # B01 -> i=B_row_offset, k=B_col_offset+1
        # B10 -> i=B_row_offset+1, k=B_col_offset
        # B11 -> i=B_row_offset+1, k=B_col_offset+1
        
        # C00 -> k=B_col_offset, i=0 -> idx k*n + i
        # C01 -> k=B_col_offset+1, i=0 -> idx k*n + i
        # C10 -> k=B_col_offset, i=1 -> idx k*n + i
        # C11 -> k=B_col_offset+1, i=1 -> idx k*n + i
        
        uA00 = 0 * m + A_col_offset
        uA01 = 0 * m + A_col_offset + 1
        uA10 = 1 * m + A_col_offset
        uA11 = 1 * m + A_col_offset + 1
        
        vB00 = B_row_offset * p + B_col_offset
        vB01 = B_row_offset * p + B_col_offset + 1
        vB10 = (B_row_offset + 1) * p + B_col_offset
        vB11 = (B_row_offset + 1) * p + B_col_offset + 1
        
        wC00 = B_col_offset * n + 0
        wC01 = (B_col_offset + 1) * n + 0
        wC10 = B_col_offset * n + 1
        wC11 = (B_col_offset + 1) * n + 1
        
        # M1
        add_term([(uA00, 1), (uA11, 1)], [(vB00, 1), (vB11, 1)], [(wC00, 1), (wC11, 1)], 1)
        # M2
        add_term([(uA10, 1), (uA11, 1)], [(vB00, 1)], [(wC10, 1)], 1)
        # M3
        add_term([(uA00, 1)], [(vB10, 1), (vB11, -1)], [(wC01, 1)], 1)
        # M4
        add_term([(uA11, 1)], [(vB01, 1), (vB00, -1)], [(wC00, 1), (wC10, 1)], 1)
        # M5
        add_term([(uA00, 1), (uA01, 1)], [(vB11, 1)], [(wC00, -1), (wC01, 1)], 1)
        # M6
        add_term([(uA10, 1), (uA00, -1)], [(vB00, 1), (vB01, 1)], [(wC11, 1)], 1)
        # M7
        add_term([(uA01, 1), (uA11, -1)], [(vB10, 1), (vB11, 1)], [(wC00, 1)], 1)

    # Instance 1: B cols 0, 1
    add_strassen(0, 0, 0)
    # Instance 2: B cols 2, 3
    add_strassen(0, 0, 2)
    
    return {"U": U, "V": V, "W": W}
