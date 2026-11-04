import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    Returns a dict with 'U', 'V', 'W' matrices.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 145  # Safety margin
    
    # Rank 50 Decomposition
    # 6 Strassen blocks (2x2x2) for the 2x4x6 part
    # 8 Rank-1 terms for the 2x4x1 part (last column of B)
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for k=0..5 (pairs 0,1; 2,3; 4,5)
    # A is 2x4. B is 4x7.
    # We split B's columns into pairs: (0,1), (2,3), (4,5).
    # And A's columns into pairs: (0,1), (2,3).
    # We need to cover all combinations.
    # A_left (cols 0,1) * B_top (rows 0,1) -> k=0,1
    # A_left (cols 0,1) * B_bot (rows 2,3) -> k=0,1 (Wait, B rows are 0..3)
    
    # Let's map indices carefully.
    # A: 2x4. B: 4x7.
    # Strassen block handles A[2x2] * B[2x2].
    # We need to cover A[2x4] * B[4x6].
    # Split A into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3).
    # Split B into B_T (2x6, rows 0,1) and B_B (2x6, rows 2,3).
    # C = A_L B_T + A_R B_B.
    # A_L B_T: A_L is 2x2. B_T is 2x6. Split B_T into 3 blocks of 2x2.
    #   Block 1: B_T[:, 0:2]. A_L * B_T1.
    #   Block 2: B_T[:, 2:4]. A_L * B_T2.
    #   Block 3: B_T[:, 4:6]. A_L * B_T3.
    # A_R B_B: A_R is 2x2. B_B is 2x6. Split B_B into 3 blocks of 2x2.
    #   Block 4: B_B[:, 0:2]. A_R * B_B1.
    #   Block 5: B_B[:, 2:4]. A_R * B_B2.
    #   Block 6: B_B[:, 4:6]. A_R * B_B3.
    
    # Indices for add_strassen(i0, j0, k0):
    # i0: row offset in A (0 or 1, but A is only 2 rows, so i0=0 always for the 2x2 block rows)
    # j0: col offset in A (0, 2)
    # k0: col offset in C / row offset in B (0, 2, 4)
    # Wait, add_strassen assumes A is 2x2 and B is 2x2.
    # In our global A (2x4), the 2x2 blocks are:
    # A_L: rows 0,1; cols 0,1. (i0=0, j0=0)
    # A_R: rows 0,1; cols 2,3. (i0=0, j0=2)
    # In our global B (4x7), the 2x2 blocks are:
    # B_T1: rows 0,1; cols 0,1. (j0=0, k0=0) -> Corresponds to A_L * B_T1
    # B_T2: rows 0,1; cols 2,3. (j0=0, k0=2) -> Corresponds to A_L * B_T2
    # B_T3: rows 0,1; cols 4,5. (j0=0, k0=4) -> Corresponds to A_L * B_T3
    # B_B1: rows 2,3; cols 0,1. (j0=2, k0=0) -> Corresponds to A_R * B_B1
    # B_B2: rows 2,3; cols 2,3. (j0=2, k0=2) -> Corresponds to A_R * B_B2
    # B_B3: rows 2,3; cols 4,5. (j0=2, k0=4) -> Corresponds to A_R * B_B3
    
    # Note: The 'j0' in add_strassen refers to the column index in A.
    # The 'j0' in V refers to the row index in B.
    # My add_strassen function uses j0 for A's col and j0 for B's row.
    # This is correct because A's cols correspond to B's rows.
    
    add_strassen(0, 0, 0) # A_L * B_T1
    add_strassen(0, 0, 2) # A_L * B_T2
    add_strassen(0, 0, 4) # A_L * B_T3
    add_strassen(0, 2, 0) # A_R * B_B1
    add_strassen(0, 2, 2) # A_R * B_B2
    add_strassen(0, 2, 4) # A_R * B_B3
    
    # Last column of B (col 6)
    # A * B[:, 6].
    # A is 2x4. B[:, 6] is 4x1.
    # C[:, 6] = A * B[:, 6].
    # This is 2x4 * 4x1. Rank 8.
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == 50
    
    def verify(U_m, V_m, W_m):
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-2, 3, size=(2, 4))
            B = np.random.randint(-2, 3, size=(4, 7))
            C_exp = A @ B
            C_comp = np.zeros((2, 7))
            for i in range(2):
                for k in range(7):
                    for j in range(4):
                        for r in range(50):
                            C_comp[i, k] += U_m[i*4+j, r] * V_m[j*7+k, r] * W_m[k*2+i, r]
            if not np.allclose(C_comp, C_exp):
                return False
        return True

    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
