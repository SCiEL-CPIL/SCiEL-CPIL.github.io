import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # n=2, m=3, p=10
    # U: 6 x R
    # V: 30 x R
    # W: 20 x R
    
    # We will build the matrices column by column.
    # Each column corresponds to one rank-1 term.
    
    # Helper to add a term
    # term_u: dict mapping (i, j) to coeff
    # term_v: dict mapping (j, k) to coeff
    # term_w: dict mapping (k, i) to coeff
    
    # We'll use lists of dicts or just sparse representations.
    # Since R is small (55), we can just build the matrices directly.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # --- Part 1: A1 * B1 (Strassen) ---
    # A1 is rows 0-1, cols 0-1 of A.
    # B1 is rows 0-1, cols 0-9 of B.
    # Split B1 into 5 blocks of 2 columns.
    
    # Strassen coefficients for 2x2x2
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # x indices: (0,0), (0,1), (1,0), (1,1) -> A rows 0,1 cols 0,1
    # y indices: (0,0), (0,1), (1,0), (1,1) -> B rows 0,1 cols 0,1 (local)
    
    # Local A indices: (0,0), (0,1), (1,0), (1,1)
    # Local B indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        { # M1
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # z11=1, z22=1
        },
        { # M2
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): 1} # z21=1, z22=1
        },
        { # M3
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1} # z12=1, z22=1
        },
        { # M4
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1} # z11=1, z21=1
        },
        { # M5
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1} # z11=-1, z12=1
        },
        { # M6
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1} # z22=1
        },
        { # M7
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): -1} # z11=1, z22=-1
        }
    ]
    
    # Note: The 'w' dict maps local (row, col) to coeff.
    # Local row 0,1 -> Global row 0,1
    # Local col 0,1 -> Global col 2*b, 2*b+1
    
    for b in range(5):
        for term in strassen_terms:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Map u
            for (ri, cj), coeff in term['u'].items():
                # A1 is at A[ri, cj]
                # Global A index: ri * m + cj
                # m=3
                global_idx = ri * m + cj
                u_dict[global_idx] = coeff
                
            # Map v
            for (ri, cj), coeff in term['v'].items():
                # B1 block b is at B[ri, 2*b + cj]
                # Global B index: ri * p + (2*b + cj)
                # p=10
                global_idx = ri * p + (2*b + cj)
                v_dict[global_idx] = coeff
                
            # Map w
            for (ri, cj), coeff in term['w'].items():
                # C block b is at C[ri, 2*b + cj]
                # Global C index: (2*b + cj) * n + ri
                # n=2
                global_idx = (2*b + cj) * n + ri
                w_dict[global_idx] = coeff
                
            # Add column
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for idx, val in u_dict.items():
                U_col[idx] = val
            for idx, val in v_dict.items():
                V_col[idx] = val
            for idx, val in w_dict.items():
                W_col[idx] = val
                
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            
    # --- Part 2: A2 * B2 (Direct) ---
    # A2 is rows 0-1, col 2 of A.
    # B2 is row 2, cols 0-9 of B.
    # Split B2 into 5 blocks of 2 columns.
    
    # A2 entries: a[0,2], a[1,2]
    # B2 block b entries: b[2, 2*b], b[2, 2*b+1]
    
    # Terms:
    # C[0, 2*b] = a[0,2] * b[2, 2*b]
    # C[0, 2*b+1] = a[0,2] * b[2, 2*b+1]
    # C[1, 2*b] = a[1,2] * b[2, 2*b]
    # C[1, 2*b+1] = a[1,2] * b[2, 2*b+1]
    
    for b in range(5):
        # Term 1: C[0, 2*b]
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        
        U_col[0 * m + 2] = 1 # a[0,2]
        V_col[2 * p + 2*b] = 1 # b[2, 2*b]
        W_col[(2*b) * n + 0] = 1 # c[0, 2*b]
        
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # Term 2: C[0, 2*b+1]
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        
        U_col[0 * m + 2] = 1 # a[0,2]
        V_col[2 * p + 2*b + 1] = 1 # b[2, 2*b+1]
        W_col[(2*b + 1) * n + 0] = 1 # c[0, 2*b+1]
        
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # Term 3: C[1, 2*b]
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        
        U_col[1 * m + 2] = 1 # a[1,2]
        V_col[2 * p + 2*b] = 1 # b[2, 2*b]
        W_col[(2*b) * n + 1] = 1 # c[1, 2*b]
        
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # Term 4: C[1, 2*b+1]
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        
        U_col[1 * m + 2] = 1 # a[1,2]
        V_col[2 * p + 2*b + 1] = 1 # b[2, 2*b+1]
        W_col[(2*b + 1) * n + 1] = 1 # c[1, 2*b+1]
        
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
    # Convert lists of columns to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
