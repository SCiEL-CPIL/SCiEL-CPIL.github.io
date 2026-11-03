import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen steps definitions
    # Each step is a tuple (A_coeff_dict, B_coeff_dict, C_targets)
    # A_coeff_dict maps (i, j) to coeff. i in {0,1}, j in {0,1,2,3}
    # B_coeff_dict maps (j, k) to coeff. j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # C_targets is a list of (block_idx, sign)
    # block_idx 0: C11 (i=0, k=0,1,2)
    # block_idx 1: C12 (i=0, k=3,4,5)
    # block_idx 2: C21 (i=1, k=0,1,2)
    # block_idx 3: C22 (i=1, k=3,4,5)
    
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0), (0,1). A22: (1,2), (1,3)
    # B11: (0,0..2), (1,0..2). B22: (2,3..5), (3,3..5)
    steps = [
        # s=0: M1
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, 
             (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        # s=1: M2 = (A21+A22)B11
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        # s=2: M3 = A11(B12-B22)
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1,
             (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        # s=3: M4 = A22(B21-B11)
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1,
             (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        # s=4: M5 = (A11+A12)B22
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        # s=5: M6 = (A21-A11)(B11+B12)
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1,
             (0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        # s=6: M7 = (A12-A22)(B21+B22)
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1,
             (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for s, (A_dict, B_dict, C_targets) in enumerate(steps):
        # The sub-multiplication is X * Y = Z
        # X is a 1x2 vector derived from A_dict
        # Y is a 2x3 matrix derived from B_dict
        # Z is a 1x3 vector derived from C_targets
        
        # We need to decompose the 1x2 * 2x3 product into 6 rank-1 terms.
        # The 6 terms correspond to:
        # 1. X[0] * Y[0,0] -> Z[0]
        # 2. X[1] * Y[1,0] -> Z[0]
        # 3. X[0] * Y[0,1] -> Z[1]
        # 4. X[1] * Y[1,1] -> Z[1]
        # 5. X[0] * Y[0,2] -> Z[2]
        # 6. X[1] * Y[1,2] -> Z[2]
        
        # We need to map the elements in A_dict to X[0] and X[1].
        # A_dict contains elements from A11, A12, A21, A22.
        # In the Strassen formula, the "X" vector is the sum/difference of these blocks.
        # For example, in M1, X = A11 + A22 = [A00+A12, A01+A13].
        # So X[0] corresponds to A elements with j=0,2? No.
        # A11 is A[0,0:2] = [A00, A01]. A22 is A[1,2:4] = [A12, A13].
        # So X[0] = A00 + A12. X[1] = A01 + A13.
        
        # Similarly for Y. Y is a 2x3 matrix.
        # In M1, Y = B11 + B22.
        # B11 is B[0:2, 0:3]. B22 is B[2:4, 3:6].
        # So Y[0,0] = B00 + B23. Y[0,1] = B01 + B24. Y[0,2] = B02 + B25.
        # Y[1,0] = B10 + B33. Y[1,1] = B11 + B34. Y[1,2] = B12 + B35.
        
        # We need to identify which A element corresponds to X[0] and X[1],
        # and which B elements correspond to Y[0,c] and Y[1,c].
        
        # The structure of the Strassen steps implies that the "X" vector elements
        # are linear combinations of specific A elements, and "Y" matrix elements
        # are linear combinations of specific B elements.
        
        # Let's extract the components.
        # For a given step, the A elements are partitioned into two groups:
        # Group 0 (for X[0]) and Group 1 (for X[1]).
        # How to determine which is which?
        # In M1: X[0] uses A00, A12. X[1] uses A01, A13.
        # Notice that A00, A01 are in row 0. A12, A13 are in row 1.
        # The indices j are 0, 1 for X[0] and X[1]? No.
        # A00 (j=0), A12 (j=2). A01 (j=1), A13 (j=3).
        # It seems X[0] uses even j? No, A00 is j=0, A12 is j=2.
        # X[1] uses odd j? A01 is j=1, A13 is j=3.
        
        # Let's check M2: X = A21 + A22 = [A10+A12, A11+A13].
        # X[0] = A10 (j=0), A12 (j=2).
        # X[1] = A11 (j=1), A13 (j=3).
        # Same pattern: X[0] uses j=0,2. X[1] uses j=1,3.
        
        # Let's check M3: X = A11 = [A00, A01].
        # X[0] = A00 (j=0). X[1] = A01 (j=1).
        # Pattern holds: X[0] uses j=0,2. X[1] uses j=1,3.
        
        # So, for any step, we can assign A elements to X[0] or X[1] based on j % 2.
        # If j % 2 == 0, it's part of X[0].
        # If j % 2 == 1, it's part of X[1].
        
        # Now for Y. Y is 2x3.
        # Y[0, c] and Y[1, c].
        # In M1: Y[0,0] = B00 (j=0), B23 (j=2).
        # Y[1,0] = B10 (j=1), B33 (j=3).
        # So Y[0, c] uses B elements with j % 2 == 0.
        # Y[1, c] uses B elements with j % 2 == 1.
        
        # What about the column c?
        # In M1: Y[0,0] uses B00 (k=0), B23 (k=3).
        # Y[0,1] uses B01 (k=1), B24 (k=4).
        # Y[0,2] uses B02 (k=2), B25 (k=5).
        # So Y[0, c] uses B elements with k = c and k = c+3.
        # Y[1, c] uses B elements with j=1,3 and k=c, c+3.
        
        # This pattern seems consistent for the Strassen steps defined.
        
        # Let's verify with M3: Y = B12 - B22.
        # B12: j=0,1; k=3,4,5.
        # B22: j=2,3; k=3,4,5.
        # Y[0, 0] (c=0) should use j=0,2 and k=0,3?
        # But B12 has k=3,4,5. B22 has k=3,4,5.
        # Wait, in M3, the output C is C12 and C22.
        # C12 is k=3,4,5. C22 is k=3,4,5.
        # So the "local" columns 0,1,2 of Y map to global columns 3,4,5.
        
        # Let's refine the mapping.
        # The "local" column index $c \in \{0,1,2\}$ of the sub-multiplication
        # maps to global column indices based on the C targets.
        
        # For M1 (C11, C22):
        # C11 is k=0,1,2. C22 is k=3,4,5.
        # The sub-product Z has 3 elements z0, z1, z2.
        # z0 maps to C11[0] (k=0) and C22[0] (k=3).
        # z1 maps to C11[1] (k=1) and C22[1] (k=4).
        # z2 maps to C11[2] (k=2) and C22[2] (k=5).
        
        # So for a given local column $c$, the global columns are determined by the C targets.
        # If C target is C11 (block 0), global k = c.
        # If C target is C12 (block 1), global k = 3 + c.
        # If C target is C21 (block 2), global k = c.
        # If C target is C22 (block 3), global k = 3 + c.
        
        # Now, which B elements contribute to Y[0, c] and Y[1, c]?
        # Y[0, c] uses B elements with j % 2 == 0.
        # Y[1, c] uses B elements with j % 2 == 1.
        # The global column of these B elements must match the global columns of the C targets?
        # No, the B elements are inputs. The C targets are outputs.
        # The B elements in the dictionary B_dict already have specific global indices (j, k).
        # We just need to group them into Y[0, c] and Y[1, c].
        
        # How to determine $c$ for a given B element (j, k) in B_dict?
        # In M1, B00 (k=0) is in Y[0, 0]. B23 (k=3) is in Y[0, 0].
        # So $c$ is determined by $k \pmod 3$?
        # B00: k=0 -> c=0. B23: k=3 -> c=0.
        # B01: k=1 -> c=1. B24: k=4 -> c=1.
        # B02: k=2 -> c=2. B25: k=5 -> c=2.
        # Yes, $c = k \pmod 3$.
        
        # So the algorithm for each step is:
        # 1. Partition A elements into X0 (j%2==0) and X1 (j%2==1).
        # 2. Partition B elements into 6 groups Y[0,c] and Y[1,c] based on j%2 and k%3.
        # 3. For each of the 6 sub-terms (x_idx, y_row, c):
        #    - x_idx is 0 or 1.
        #    - y_row is 0 or 1.
        #    - c is 0, 1, or 2.
        #    - The term is X[x_idx] * Y[y_row, c] -> Z[c].
        #    - U gets the coefficients from the A elements in X[x_idx].
        #    - V gets the coefficients from the B elements in Y[y_row, c].
        #    - W gets the coefficients for the C targets at local column c.
        
        # Let's implement this.
        
        X0 = {k: v for k, v in A_dict.items() if k[1] % 2 == 0}
        X1 = {k: v for k, v in A_dict.items() if k[1] % 2 == 1}
        
        # Y[0, c] and Y[1, c]
        Y = [[[], [], []], [[], [], []]]
        for (j, k), coeff in B_dict.items():
            row = j % 2
            col = k % 3
            Y[row][col].append(((j, k), coeff))
        
        sub_terms = [
            (X0, Y[0][0], 0),
            (X1, Y[1][0], 0),
            (X0, Y[0][1], 1),
            (X1, Y[1][1], 1),
            (X0, Y[0][2], 2),
            (X1, Y[1][2], 2),
        ]
        
        for X_c, Y_c, c_idx in sub_terms:
            # U
            for (i, j), coeff in X_c.items():
                U[i * m + j, col] += coeff
            
            # V
            for (j, k), coeff in Y_c:
                V[j * p + k, col] += coeff
            
            # W
            for block_idx, sign in C_targets:
                if block_idx == 0: # C11
                    i_C = 0
                    k_C = c_idx
                elif block_idx == 1: # C12
                    i_C = 0
                    k_C = 3 + c_idx
                elif block_idx == 2: # C21
                    i_C = 1
                    k_C = c_idx
                elif block_idx == 3: # C22
                    i_C = 1
                    k_C = 3 + c_idx
                
                W[k_C * n + i_C, col] += sign
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verification
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-6:
        # Fallback to schoolbook
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
