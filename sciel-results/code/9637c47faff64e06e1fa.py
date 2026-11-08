import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank 60 decomposition
    # 2x4 * 4x8 using 2x(2x2 * 2x8) -> Rank 28
    # 1x4 * 4x8 using trivial -> Rank 32
    # Total Rank 60
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2 decomposition
    # u_offset: row offset in U (i*m + j)
    # v_offset: row offset in V (j*p + k)
    # w_offset: row offset in W (k*n + i)
    # The 2x2 block is at i0, j0 for U; j0, k0 for V; k0, i0 for W
    def add_strassen_2x2(u_i0, u_j0, v_j0, v_k0, w_k0, w_i0, col_start):
        # M1
        U[u_i0 * m + u_j0][col_start] = 1
        U[(u_i0+1) * m + (u_j0+1)][col_start] = 1
        V[v_j0 * p + v_k0][col_start] = 1
        V[(v_j0+1) * p + (v_k0+1)][col_start] = 1
        W[w_k0 * n + w_i0][col_start] = 1
        W[(w_k0+1) * n + (w_i0+1)][col_start] = 1
        col_start += 1
        
        # M2
        U[(u_i0+1) * m + u_j0][col_start] = 1
        U[(u_i0+1) * m + (u_j0+1)][col_start] = 1
        V[v_j0 * p + v_k0][col_start] = 1
        W[(w_k0+1) * n + w_i0][col_start] = 1
        W[(w_k0+1) * n + (w_i0+1)][col_start] = 1
        col_start += 1
        
        # M3
        U[u_i0 * m + u_j0][col_start] = 1
        V[v_j0 * p + (v_k0+1)][col_start] = 1
        V[(v_j0+1) * p + (v_k0+1)][col_start] = -1
        W[w_k0 * n + (w_i0+1)][col_start] = 1
        col_start += 1
        
        # M4
        U[(u_i0+1) * m + (u_j0+1)][col_start] = 1
        V[(v_j0+1) * p + v_k0][col_start] = 1
        V[v_j0 * p + v_k0][col_start] = -1
        W[w_k0 * n + w_i0][col_start] = 1
        W[(w_k0+1) * n + w_i0][col_start] = 1
        col_start += 1
        
        # M5
        U[u_i0 * m + u_j0][col_start] = 1
        U[u_i0 * m + (u_j0+1)][col_start] = 1
        V[(v_j0+1) * p + (v_k0+1)][col_start] = 1
        W[w_k0 * n + w_i0][col_start] = 1
        W[w_k0 * n + (w_i0+1)][col_start] = 1
        col_start += 1
        
        # M6
        U[(u_i0+1) * m + u_j0][col_start] = 1
        U[u_i0 * m + u_j0][col_start] = -1
        V[v_j0 * p + v_k0][col_start] = 1
        V[v_j0 * p + (v_k0+1)][col_start] = 1
        W[(w_k0+1) * n + (w_i0+1)][col_start] = 1
        col_start += 1
        
        # M7
        U[u_i0 * m + (u_j0+1)][col_start] = 1
        U[(u_i0+1) * m + (u_j0+1)][col_start] = -1
        V[(v_j0+1) * p + v_k0][col_start] = 1
        V[(v_j0+1) * p + (v_k0+1)][col_start] = 1
        W[w_k0 * n + w_i0][col_start] = 1
        W[(w_k0+1) * n + (w_i0+1)][col_start] = 1
        col_start += 1
        
        return col_start

    # Part 1: 2x4 * 4x8
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # A1 is i=0,1; j=0,1
    # A2 is i=0,1; j=2,3
    # B is 4x8
    # B1 is j=0,1; k=0..7
    # B2 is j=2,3; k=0..7
    
    # A1 * B1
    # B1 is 2x8. Split into B11 (2x4) and B12 (2x4)?
    # No, 2x2 * 2x8.
    # Split 2x8 into two 2x4?
    # 2x2 * 2x4.
    # Split 2x4 into two 2x2.
    # So 2x2 * 2x8 requires 4 Strassen 2x2 blocks?
    # No.
    # 2x2 * 2x8.
    # C = A B.
    # A is 2x2. B is 2x8.
    # We can compute A * B[:, 0:4] and A * B[:, 4:8].
    # Each is 2x2 * 2x4.
    # 2x2 * 2x4 can be done with 2 Strassen 2x2 blocks?
    # 2x2 * 2x4.
    # Split B (2x4) into B1 (2x2) and B2 (2x2).
    # C1 = A B1 (2x2 * 2x2). Rank 7.
    # C2 = A B2 (2x2 * 2x2). Rank 7.
    # Total 14.
    
    # So for A1 (2x2) * B1 (2x8):
    # B1 is rows 0,1 of B.
    # Split B1 into B1a (cols 0-3) and B1b (cols 4-7).
    # B1a is 2x4. Split into B1a1 (cols 0-1) and B1a2 (cols 2-3).
    # B1a1 is 2x2.
    
    # A1 * B1a1 (2x2 * 2x2)
    # i=0,1; j=0,1; k=0,1
    col = add_strassen_2x2(0, 0, 0, 0, 0, 0, col)
    
    # A1 * B1a2 (2x2 * 2x2)
    # i=0,1; j=0,1; k=2,3
    col = add_strassen_2x2(0, 0, 0, 2, 2, 0, col)
    
    # A1 * B1b1 (2x2 * 2x2)
    # i=0,1; j=0,1; k=4,5
    col = add_strassen_2x2(0, 0, 0, 4, 4, 0, col)
    
    # A1 * B1b2 (2x2 * 2x2)
    # i=0,1; j=0,1; k=6,7
    col = add_strassen_2x2(0, 0, 0, 6, 6, 0, col)
    
    # A2 * B2
    # A2 is i=0,1; j=2,3
    # B2 is j=2,3; k=0..7
    # Split B2 into B2a (k=0-3) and B2b (k=4-7).
    # B2a into B2a1 (k=0,1) and B2a2 (k=2,3).
    
    # A2 * B2a1 (2x2 * 2x2)
    # i=0,1; j=2,3; k=0,1
    col = add_strassen_2x2(0, 2, 2, 0, 0, 0, col)
    
    # A2 * B2a2 (2x2 * 2x2)
    # i=0,1; j=2,3; k=2,3
    col = add_strassen_2x2(0, 2, 2, 2, 2, 0, col)
    
    # A2 * B2b1 (2x2 * 2x2)
    # i=0,1; j=2,3; k=4,5
    col = add_strassen_2x2(0, 2, 2, 4, 4, 0, col)
    
    # A2 * B2b2 (2x2 * 2x2)
    # i=0,1; j=2,3; k=6,7
    col = add_strassen_2x2(0, 2, 2, 6, 6, 0, col)
    
    # Part 2: 1x4 * 4x8
    # A3 is i=2; j=0..3
    # B is j=0..3; k=0..7
    # Trivial decomposition
    for j in range(4):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
