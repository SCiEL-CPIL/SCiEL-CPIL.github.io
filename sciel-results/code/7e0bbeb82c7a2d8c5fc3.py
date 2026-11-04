import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off, c_off):
        nonlocal col
        # i_off: row offset in A and C
        # j_off: col offset in A, row offset in B
        # k_off: col offset in B and C
        
        # Local indices
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # Global indices
        A_idx = [
            (i_off + 0) * m + (j_off + 0),
            (i_off + 0) * m + (j_off + 1),
            (i_off + 1) * m + (j_off + 0),
            (i_off + 1) * m + (j_off + 1)
        ]
        B_idx = [
            (j_off + 0) * p + (k_off + 0),
            (j_off + 0) * p + (k_off + 1),
            (j_off + 1) * p + (k_off + 0),
            (j_off + 1) * p + (k_off + 1)
        ]
        C_idx = [
            (k_off + 0) * n + (i_off + 0),
            (k_off + 0) * n + (i_off + 1),
            (k_off + 1) * n + (i_off + 0),
            (k_off + 1) * n + (i_off + 1)
        ]
        
        # Strassen terms
        # M1 = (a+b)(e+h)
        U[A_idx[0], col] += 1
        U[A_idx[1], col] += 1
        V[B_idx[0], col] += 1
        V[B_idx[3], col] += 1
        W[C_idx[0], col] += 1 # C11
        W[C_idx[3], col] += 1 # C22
        col += 1
        
        # M2 = (c+d)h
        U[A_idx[2], col] += 1
        U[A_idx[3], col] += 1
        V[B_idx[3], col] += 1
        W[C_idx[2], col] += 1 # C21
        W[C_idx[3], col] -= 1 # C22
        col += 1
        
        # M3 = a(f-h)
        U[A_idx[0], col] += 1
        V[B_idx[1], col] += 1
        V[B_idx[3], col] -= 1
        W[C_idx[1], col] += 1 # C12
        W[C_idx[3], col] += 1 # C22
        col += 1
        
        # M4 = d(g-e)
        U[A_idx[3], col] += 1
        V[B_idx[2], col] += 1
        V[B_idx[0], col] -= 1
        W[C_idx[0], col] += 1 # C11
        W[C_idx[2], col] += 1 # C21
        col += 1
        
        # M5 = (a+d)(e+f)
        U[A_idx[0], col] += 1
        U[A_idx[3], col] += 1
        V[B_idx[0], col] += 1
        V[B_idx[1], col] += 1
        W[C_idx[0], col] -= 1 # C11
        W[C_idx[1], col] += 1 # C12
        col += 1
        
        # M6 = (b+d)(g+f)
        U[A_idx[1], col] += 1
        U[A_idx[3], col] += 1
        V[B_idx[2], col] += 1
        V[B_idx[1], col] += 1
        W[C_idx[3], col] += 1 # C22
        col += 1
        
        # M7 = (a-c)(e+g)
        U[A_idx[0], col] += 1
        U[A_idx[2], col] -= 1
        V[B_idx[0], col] += 1
        V[B_idx[2], col] += 1
        W[C_idx[0], col] += 1 # C11
        col += 1

    # Add 4 Strassen blocks for i in {0,1}
    for i_off in [0]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                add_strassen(i_off, j_off, k_off)

    # Add schoolbook block for i=2
    # i_off = 2, j in 0..3, k in 0..3
    # U[row 2, col j], V[row j, col k], W[col k, row 2]
    for j in range(4):
        for k in range(4):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
