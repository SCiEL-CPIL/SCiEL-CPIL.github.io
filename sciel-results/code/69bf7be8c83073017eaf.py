"""Matrix multiplication M(2,3,10) via block decomposition: Strassen for M(2,2,10) + outer product for M(2,1,10)."""
DIMENSIONS = [2, 3, 10]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)
    
    # Index conventions:
    # U[i*m + j, r] where i in [0,1], j in [0,2]
    # V[j*p + k, r] where j in [0,2], k in [0,9]
    # W[k*n + i, r] where k in [0,9], i in [0,1]
    
    # Part 1: Strassen decomposition for the top-left 2x2 block of A times top 2 rows of B
    # A_top = A[0:2, 0:2] (2x2), B_top = B[0:2, :] (2x10)
    # This is 5 independent 2x2x2 multiplications (one for each pair of columns in B)
    
    # For each block b in 0..4, covering columns 2b and 2b+1 of B:
    # A_top is 2x2, B_block is 2x2 (columns 2b, 2b+1 of B_top)
    # C_block is 2x2 (columns 2b, 2b+1 of C_top)
    
    # Strassen for M(2,2,2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A_top indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # In global U: row i*m+j where i=0,1 (top 2 rows of A), j=0,1 (first 2 cols)
    # So A11 -> U[0*3+0] = U[0], A12 -> U[0*3+1] = U[1], A21 -> U[1*3+0] = U[3], A22 -> U[1*3+1] = U[4]
    
    # B_top indices: B[0,k], B[1,k] for k=0..9
    # In global V: row j*p+k where j=0,1 (first 2 rows of B), k=0..9
    # So B[0,k] -> V[0*10+k] = V[k], B[1,k] -> V[1*10+k] = V[10+k]
    
    # C_top indices: C[i,k] for i=0,1, k=0..9
    # In global W: row k*n+i where k=0..9, i=0,1
    # So C[0,k] -> W[k*2+0] = W[2k], C[1,k] -> W[k*2+1] = W[2k+1]
    
    # Strassen coefficients for each block
    # For block b, columns 2b and 2b+1:
    # B11 = B[0, 2b] -> V[2b]
    # B12 = B[0, 2b+1] -> V[2b+1]
    # B21 = B[1, 2b] -> V[10+2b]
    # B22 = B[1, 2b+1] -> V[10+2b+1]
    
    # A11 -> U[0], A12 -> U[1], A21 -> U[3], A22 -> U[4]
    
    # For each Strassen multiplication r in 1..7, and each block b in 0..4:
    # We add to U, V, W the appropriate coefficients
    
    # Define Strassen factors:
    # Each factor r has:
    #   A_coeffs: dict from A_index (0=A11, 1=A12, 2=A21, 3=A22) to coefficient
    #   B_coeffs: dict from B_index (0=B11, 1=B12, 2=B21, 3=B22) to coefficient
    
    A_idx = {'A11': 0, 'A12': 1, 'A21': 3, 'A22': 4}  # global U row indices
    B_idx = {'B11': 0, 'B12': 1, 'B21': 10, 'B22': 11}  # offsets within block
    
    strassen_factors = [
        # (A_coeffs, B_coeffs)
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}),  # M1
        ({'A21': 1, 'A22': 1}, {'B11': 1}),             # M2
        ({'A11': 1}, {'B12': 1, 'B22': -1}),            # M3
        ({'A22': 1}, {'B21': 1, 'B11': -1}),            # M4
        ({'A11': 1, 'A12': 1}, {'B22': 1}),             # M5
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}),  # M6
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}),  # M7
    ]
    
    # C combination: C11, C12, C21, C22 in terms of M1..M7
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block b and each Strassen factor r, we create one column.
    # Total Strassen columns: 5 blocks * 7 factors = 35
    
    col = 0
    for b in range(5):
        for r in range(7):
            A_c, B_c = strassen_factors[r]
            
            # Set U coefficients
            for name, coeff in A_c.items():
                U[A_idx[name], col] += coeff
            
            # Set V coefficients
            for name, coeff in B_c.items():
                base = B_idx[name]
                # For block b, B11 is at column 2b, B12 at 2b+1, etc.
                # B11 offset 0 -> column 2b, B12 offset 1 -> column 2b+1
                # B21 offset 10 -> column 10+2b, B22 offset 11 -> column 10+2b+1
                if name == 'B11':
                    V[2*b, col] += coeff
                elif name == 'B12':
                    V[2*b+1, col] += coeff
                elif name == 'B21':
                    V[10+2*b, col] += coeff
                elif name == 'B22':
                    V[10+2*b+1, col] += coeff
            
            # Set W coefficients
            # C11 = M1 + M4 - M5 + M7 -> W[2b] (C[0, 2b])
            # C12 = M3 + M5 -> W[2b+1] (C[0, 2b+1])
            # C21 = M2 + M4 -> W[2b+2] (C[1, 2b])
            # C22 = M1 - M2 + M3 + M6 -> W[2b+3] (C[1, 2b+1])
            
            # M1 (r=0): C11 += 1, C22 += 1
            # M2 (r=1): C21 += 1, C22 -= 1
            # M3 (r=2): C12 += 1, C22 += 1
            # M4 (r=3): C11 += 1, C21 += 1
            # M5 (r=4): C11 -= 1, C12 += 1
            # M6 (r=5): C22 += 1
            # M7 (r=6): C11 += 1
            
            if r == 0:  # M1
                W[2*b, col] += 1      # C11
                W[2*b+3, col] += 1    # C22
            elif r == 1:  # M2
                W[2*b+2, col] += 1    # C21
                W[2*b+3, col] -= 1    # C22
            elif r == 2:  # M3
                W[2*b+1, col] += 1    # C12
                W[2*b+3, col] += 1    # C22
            elif r == 3:  # M4
                W[2*b, col] += 1      # C11
                W[2*b+2, col] += 1    # C21
            elif r == 4:  # M5
                W[2*b, col] -= 1      # C11
                W[2*b+1, col] += 1    # C12
            elif r == 5:  # M6
                W[2*b+3, col] += 1    # C22
            elif r == 6:  # M7
                W[2*b, col] += 1      # C11
            
            col += 1
    
    # Part 2: Outer product for A[0:2, 2] (2x1) times B[2, :] (1x10)
    # A_bottom_row = A[:, 2] (2x1), B_bottom_row = B[2, :] (1x10)
    # C_bottom = A_bottom_row @ B_bottom_row (2x10)
    # This requires 2*10 = 20 multiplications
    
    # A[:, 2] -> U rows: i*3+2 for i=0,1 -> U[2], U[5]
    # B[2, :] -> V rows: 2*10+k = 20+k for k=0..9
    # C[:, k] -> W rows: k*2+i for i=0,1, k=0..9
    
    for i in range(2):
        for k in range(10):
            U[i*3+2, col] = 1
            V[20+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
