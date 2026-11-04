import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=4, m=5, p=5
    # U: (20, rank), V: (25, rank), W: (20, rank)
    
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(col, U_idx, U_val, V_idx, V_val, W_idx, W_val):
        """
        Adds a rank-1 term to the decomposition.
        U_idx: list of indices in U (flattened i*m+j)
        U_val: list of values corresponding to U_idx
        V_idx: list of indices in V (flattened j*p+k)
        V_val: list of values corresponding to V_idx
        W_idx: list of indices in W (flattened k*n+i)
        W_val: list of values corresponding to W_idx
        """
        for idx, val in zip(U_idx, U_val):
            U[idx, col] += val
        for idx, val in zip(V_idx, V_val):
            V[idx, col] += val
        for idx, val in zip(W_idx, W_val):
            W[idx, col] += val

    def get_A_idx(i, j):
        return i * m + j

    def get_B_idx(j, k):
        return j * p + k

    def get_W_idx(k, i):
        return k * n + i

    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0):
        nonlocal col
        # A indices: (A_i0, A_j0), (A_i0, A_j0+1), (A_i0+1, A_j0), (A_i0+1, A_j0+1)
        # B indices: (B_j0, B_k0), (B_j0, B_k0+1), (B_j0+1, B_k0), (B_j0+1, B_k0+1)
        # C indices: (C_i0, C_k0), (C_i0, C_k0+1), (C_i0+1, C_k0), (C_i0+1, C_k0+1)
        
        a0 = get_A_idx(A_i0, A_j0)
        a1 = get_A_idx(A_i0, A_j0+1)
        a2 = get_A_idx(A_i0+1, A_j0)
        a3 = get_A_idx(A_i0+1, A_j0+1)
        
        b0 = get_B_idx(B_j0, B_k0)
        b1 = get_B_idx(B_j0, B_k0+1)
        b2 = get_B_idx(B_j0+1, B_k0)
        b3 = get_B_idx(B_j0+1, B_k0+1)
        
        c0 = get_W_idx(C_k0, C_i0)
        c1 = get_W_idx(C_k0+1, C_i0)
        c2 = get_W_idx(C_k0, C_i0+1)
        c3 = get_W_idx(C_k0+1, C_i0+1)
        
        # M1 = (a0 + a3)(b0 + b3) -> c0(+1), c3(+1)
        add_term(col, [a0, a3], [1, 1], [b0, b3], [1, 1], [c0, c3], [1, 1])
        col += 1
        
        # M2 = (a2 + a3)b0 -> c2(+1), c3(-1)
        add_term(col, [a2, a3], [1, 1], [b0], [1], [c2, c3], [1, -1])
        col += 1
        
        # M3 = a0(b1 - b3) -> c1(+1), c3(+1)
        add_term(col, [a0], [1], [b1, b3], [1, -1], [c1, c3], [1, 1])
        col += 1
        
        # M4 = a3(b2 - b0) -> c0(+1), c2(+1)
        add_term(col, [a3], [1], [b2, b0], [1, -1], [c0, c2], [1, 1])
        col += 1
        
        # M5 = (a0 + a1)b3 -> c0(-1), c1(+1)
        add_term(col, [a0, a1], [1, 1], [b3], [1], [c0, c1], [-1, 1])
        col += 1
        
        # M6 = (a2 - a0)(b0 + b1) -> c3(+1)
        add_term(col, [a2, a0], [1, -1], [b0, b1], [1, 1], [c3], [1])
        col += 1
        
        # M7 = (a1 - a3)(b2 + b3) -> c0(+1)
        add_term(col, [a1, a3], [1, -1], [b2, b3], [1, 1], [c0], [1])
        col += 1

    # Helper to add a Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(A_i0, A_j0, B_j0, B_k0, C_i0, C_k0):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        # A indices: (A_i0, A_j0), (A_i0, A_j0+1), (A_i0+1, A_j0), (A_i0+1, A_j0+1)
        # B indices: (B_j0, B_k0), (B_j0+1, B_k0)  -- Note: B is 2x1, so k is fixed?
        # Wait, in our block structure, B_{02} is 2x1.
        # B_{02} corresponds to j in {0,1}, k=4.
        # So B indices are (0,4) and (1,4).
        # C indices: (C_k0, C_i0), (C_k0, C_i0+1).
        
        a0 = get_A_idx(A_i0, A_j0)
        a1 = get_A_idx(A_i0, A_j0+1)
        a2 = get_A_idx(A_i0+1, A_j0)
        a3 = get_A_idx(A_i0+1, A_j0+1)
        
        b0 = get_B_idx(B_j0, B_k0)
        b1 = get_B_idx(B_j0+1, B_k0)
        
        c0 = get_W_idx(C_k0, C_i0)
        c1 = get_W_idx(C_k0, C_i0+1)
        
        # c0 = a0*b0 + a1*b1
        # c1 = a2*b0 + a3*b1
        
        # Term 1: a0 b0 c0
        add_term(col, [a0], [1], [b0], [1], [c0], [1])
        col += 1
        
        # Term 2: a1 b1 c0
        add_term(col, [a1], [1], [b1], [1], [c0], [1])
        col += 1
        
        # Term 3: a2 b0 c1
        add_term(col, [a2], [1], [b0], [1], [c1], [1])
        col += 1
        
        # Term 4: a3 b1 c1
        add_term(col, [a3], [1], [b1], [1], [c1], [1])
        col += 1

    # Apply to the 4x5x5 problem
    
    # Block C00 (i=0,1; k=0,1)
    # A00 (i=0,1; j=0,1) x B00 (j=0,1; k=0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A01 (i=0,1; j=2,3) x B10 (j=2,3; k=0,1)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block C01 (i=0,1; k=2,3)
    # A00 (i=0,1; j=0,1) x B01 (j=0,1; k=2,3)
    add_strassen(0, 0, 0, 2, 0, 2)
    # A01 (i=0,1; j=2,3) x B11 (j=2,3; k=2,3)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block C10 (i=2,3; k=0,1)
    # A10 (i=2,3; j=0,1) x B00 (j=0,1; k=0,1)
    add_strassen(2, 0, 0, 0, 2, 0)
    # A11 (i=2,3; j=2,3) x B10 (j=2,3; k=0,1)
    add_strassen(2, 2, 2, 0, 2, 0)
    
    # Block C11 (i=2,3; k=2,3)
    # A10 (i=2,3; j=0,1) x B01 (j=0,1; k=2,3)
    add_strassen(2, 0, 0, 2, 2, 2)
    # A11 (i=2,3; j=2,3) x B11 (j=2,3; k=2,3)
    add_strassen(2, 2, 2, 2, 2, 2)
    
    # Block C02 (i=0,1; k=4)
    # A00 (i=0,1; j=0,1) x B02 (j=0,1; k=4)
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    # A01 (i=0,1; j=2,3) x B12 (j=2,3; k=4)
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)
    
    # Block C12 (i=2,3; k=4)
    # A10 (i=2,3; j=0,1) x B02 (j=0,1; k=4)
    add_schoolbook_2x2x1(2, 0, 0, 4, 2, 4)
    # A11 (i=2,3; j=2,3) x B12 (j=2,3; k=4)
    add_schoolbook_2x2x1(2, 2, 2, 4, 2, 4)

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
