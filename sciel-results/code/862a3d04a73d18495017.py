# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Calculate required rank
    # Bottom part (row 2 of A): 1 x 4 x 8 -> Rank 32
    rank_bottom = 1 * m * p
    
    # Top part (rows 0,1 of A): 2 x 4 x 8
    # Using Strassen for 2x2x2 blocks.
    # A is 2x4, B is 4x8.
    # Split A into 2x2 blocks (cols 0-1, 2-3).
    # Split B into 2x8 blocks (rows 0-1, 2-3).
    # 2 blocks of 2x2x8.
    # 2x2x8 is split into two 2x2x4.
    # 2x2x4 is split into two 2x2x2.
    # 2x2x2 uses Strassen (rank 7).
    # So 2x2x8 rank = 2 * 2 * 7 = 28.
    # Total top rank = 2 * 28 = 56.
    
    total_rank = rank_bottom + 56
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        def A_idx(r, c): return (i0 + r) * m + (j0 + c)
        def B_idx(r, c): return (j0 + r) * p + (k0 + c)
        def C_idx(c, r): return (k0 + c) * n + (i0 + r)
        
        # T1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
        U[A_idx(0,0), col] = 1; U[A_idx(1,1), col] = 1
        V[B_idx(0,0), col] = 1; V[B_idx(1,1), col] = 1
        W[C_idx(0,0), col] = 1; W[C_idx(1,1), col] = 1
        col += 1
        
        # T2: (a10+a11)b00 -> c10+=1, c11-=1
        U[A_idx(1,0), col] = 1; U[A_idx(1,1), col] = 1
        V[B_idx(0,0), col] = 1
        W[C_idx(0,1), col] = 1; W[C_idx(1,1), col] = -1
        col += 1
        
        # T3: a00(b01-b11) -> c01+=1, c11+=1
        U[A_idx(0,0), col] = 1
        V[B_idx(0,1), col] = 1; V[B_idx(1,1), col] = -1
        W[C_idx(1,0), col] = 1; W[C_idx(1,1), col] = 1
        col += 1
        
        # T4: a11(b10-b00) -> c00+=1, c10+=1
        U[A_idx(1,1), col] = 1
        V[B_idx(1,0), col] = 1; V[B_idx(0,0), col] = -1
        W[C_idx(0,0), col] = 1; W[C_idx(0,1), col] = 1
        col += 1
        
        # T5: (a00+a01)b11 -> c00-=1, c01+=1
        U[A_idx(0,0), col] = 1; U[A_idx(0,1), col] = 1
        V[B_idx(1,1), col] = 1
        W[C_idx(0,0), col] = -1; W[C_idx(1,0), col] = 1
        col += 1
        
        # T6: (a10-a00)(b00+b01) -> c01+=1, c11+=1
        U[A_idx(1,0), col] = 1; U[A_idx(0,0), col] = -1
        V[B_idx(0,0), col] = 1; V[B_idx(0,1), col] = 1
        W[C_idx(1,0), col] = 1; W[C_idx(1,1), col] = 1
        col += 1
        
        # T7: (a01-a11)(b10+b11) -> c00+=1, c10-=1
        U[A_idx(0,1), col] = 1; U[A_idx(1,1), col] = -1
        V[B_idx(1,0), col] = 1; V[B_idx(1,1), col] = 1
        W[C_idx(0,0), col] = 1; W[C_idx(0,1), col] = -1
        col += 1

    # Bottom part: Row 2 of A
    for j in range(m):
        for k in range(p):
            add_term(2 * m + j, j * p + k, k * n + 2, 1.0)
    
    # Top part: Rows 0,1 of A
    # Block 1: i in [0,1], j in [0,1]
    for k0 in range(0, p, 2):
        add_strassen_block(0, 0, k0)
        
    # Block 2: i in [0,1], j in [2,3]
    for k0 in range(0, p, 2):
        add_strassen_block(0, 2, k0)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
