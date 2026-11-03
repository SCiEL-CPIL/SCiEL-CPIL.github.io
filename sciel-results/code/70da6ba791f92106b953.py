"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 10.0
    
    n, m, p = DIMENSIONS
    
    # Target rank 52
    # 4 * Strassen (2x2x2) + 2 * Schoolbook (2x2x3)
    # Strassen rank 7, Schoolbook rank 12
    # Total 4*7 + 2*12 = 28 + 24 = 52
    
    rank = 52
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a local decomposition to global matrices
    # local_n, local_m, local_p
    # i_offset, j_offset, k_offset
    # local_u, local_v, local_w (shapes local_n*local_m, local_rank, etc)
    def add_decomp(local_u, local_v, local_w, i_off, j_off, k_off, local_n, local_m, local_p):
        nonlocal col
        local_rank = local_u.shape[1]
        for r in range(local_rank):
            # U
            for i in range(local_n):
                for j in range(local_m):
                    val = local_u[i * local_m + j, r]
                    if val != 0:
                        U[(i + i_off) * m + (j + j_off), col] += val
            # V
            for j in range(local_m):
                for k in range(local_p):
                    val = local_v[j * local_p + k, r]
                    if val != 0:
                        V[(j + j_off) * p + (k + k_off), col] += val
            # W
            for k in range(local_p):
                for i in range(local_n):
                    val = local_w[k * local_n + i, r]
                    if val != 0:
                        W[(k + k_off) * n + (i + i_off), col] += val
            col += 1

    # Strassen 2x2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    def strassen_222():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        u[0, 0] = 1; u[3, 0] = 1 # A11, A22
        v[0, 0] = 1; v[3, 0] = 1 # B11, B22
        w[0, 0] = 1; w[3, 0] = 1 # C11, C22 (idx: k*n+i)
        
        # M2 = (A21+A22)B11
        u[2, 1] = 1; u[3, 1] = 1 # A21, A22
        v[0, 1] = 1 # B11
        w[1, 1] = 1; w[3, 1] = 1 # C21, C22 (C21 is k=1,i=2 -> 1*2+2? No. 
        # W index k*n+i.
        # C11: k=0, i=0 -> 0
        # C12: k=1, i=0 -> 2
        # C21: k=0, i=1 -> 1
        # C22: k=1, i=1 -> 3
        # M2 contributes to C21 and C22?
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M2 contributes +1 to C21 and -1 to C22.
        w[1, 1] = 1 # C21
        w[3, 1] = -1 # C22
        
        # M3 = A11(B21-B22)
        u[0, 2] = 1 # A11
        v[2, 2] = 1; v[3, 2] = -1 # B21, B22
        # C12 = M3 + M5
        # C22 = M1 - M2 + M3 + M6
        w[2, 2] = 1 # C12
        w[3, 2] = 1 # C22
        
        # M4 = A22(B12-B11)
        u[3, 3] = 1 # A22
        v[1, 3] = 1; v[0, 3] = -1 # B12, B11
        # C11 = M1 + M4 - M5 + M7
        # C21 = M2 + M4
        w[0, 3] = 1 # C11
        w[1, 3] = 1 # C21
        
        # M5 = (A11+A12)B22
        u[0, 4] = 1; u[1, 4] = 1 # A11, A12
        v[3, 4] = 1 # B22
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        w[0, 4] = -1 # C11
        w[2, 4] = 1 # C12
        
        # M6 = (A21-A11)(B11+B12)
        u[2, 5] = 1; u[0, 5] = -1 # A21, A11
        v[0, 5] = 1; v[1, 5] = 1 # B11, B12
        # C22 = M1 - M2 + M3 + M6
        w[3, 5] = 1 # C22
        
        # M7 = (A12-A22)(B21+B22)
        u[1, 6] = 1; u[3, 6] = -1 # A12, A22
        v[2, 6] = 1; v[3, 6] = 1 # B21, B22
        # C11 = M1 + M4 - M5 + M7
        w[0, 6] = 1 # C11
        
        return u, v, w

    # Schoolbook 2x2x3
    # A (2x2), B (2x3)
    # C (2x3)
    def schoolbook_223():
        u = np.zeros((4, 12))
        v = np.zeros((6, 12))
        w = np.zeros((6, 12))
        
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C13 = A11 B13 + A12 B23
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        # C23 = A21 B13 + A22 B23
        
        col = 0
        # A11 B11
        u[0, col] = 1; v[0, col] = 1; w[0, col] = 1; col += 1
        # A12 B21
        u[1, col] = 1; v[3, col] = 1; w[0, col] = 1; col += 1
        # A11 B12
        u[0, col] = 1; v[1, col] = 1; w[1, col] = 1; col += 1
        # A12 B22
        u[1, col] = 1; v[4, col] = 1; w[1, col] = 1; col += 1
        # A11 B13
        u[0, col] = 1; v[2, col] = 1; w[2, col] = 1; col += 1
        # A12 B23
        u[1, col] = 1; v[5, col] = 1; w[2, col] = 1; col += 1
        # A21 B11
        u[2, col] = 1; v[0, col] = 1; w[3, col] = 1; col += 1
        # A22 B21
        u[3, col] = 1; v[3, col] = 1; w[3, col] = 1; col += 1
        # A21 B12
        u[2, col] = 1; v[1, col] = 1; w[4, col] = 1; col += 1
        # A22 B22
        u[3, col] = 1; v[4, col] = 1; w[4, col] = 1; col += 1
        # A21 B13
        u[2, col] = 1; v[2, col] = 1; w[5, col] = 1; col += 1
        # A22 B23
        u[3, col] = 1; v[5, col] = 1; w[5, col] = 1; col += 1
        
        return u, v, w

    # Assemble
    # A1 (2x2) at i=0, j=0
    # A2 (2x2) at i=0, j=2
    # B1 (2x7) at j=0, k=0
    # B2 (2x7) at j=2, k=0
    
    # B1a (2x4) at j=0, k=0
    # B1b (2x3) at j=0, k=4
    # B2a (2x4) at j=2, k=0
    # B2b (2x3) at j=2, k=4
    
    # B1a1 (2x2) at j=0, k=0
    # B1a2 (2x2) at j=0, k=2
    # B2a1 (2x2) at j=2, k=0
    # B2a2 (2x2) at j=2, k=2
    
    # 1. A1 B1a1 (Strassen)
    u, v, w = strassen_222()
    add_decomp(u, v, w, 0, 0, 0, 2, 2, 2)
    
    # 2. A1 B1a2 (Strassen)
    u, v, w = strassen_222()
    add_decomp(u, v, w, 0, 0, 2, 2, 2, 2)
    
    # 3. A1 B1b (Schoolbook)
    u, v, w = schoolbook_223()
    add_decomp(u, v, w, 0, 0, 4, 2, 2, 3)
    
    # 4. A2 B2a1 (Strassen)
    u, v, w = strassen_222()
    add_decomp(u, v, w, 0, 2, 0, 2, 2, 2)
    
    # 5. A2 B2a2 (Strassen)
    u, v, w = strassen_222()
    add_decomp(u, v, w, 0, 2, 2, 2, 2, 2)
    
    # 6. A2 B2b (Schoolbook)
    u, v, w = schoolbook_223()
    add_decomp(u, v, w, 0, 2, 4, 2, 2, 3)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
