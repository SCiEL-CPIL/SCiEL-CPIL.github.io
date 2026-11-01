"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 5 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x1 (rank 4)
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # U indices: 0, 1, 2, 3
    
    # Process 5 blocks of 2x2
    for b in range(5):
        base_col = 2 * b
        # V indices for B block
        # e = B[0, base_col] -> V[0*p + base_col]
        # f = B[0, base_col+1] -> V[0*p + base_col+1]
        # g = B[1, base_col] -> V[1*p + base_col]
        # h = B[1, base_col+1] -> V[1*p + base_col+1]
        
        v_e = 0 * p + base_col
        v_f = 0 * p + base_col + 1
        v_g = 1 * p + base_col
        v_h = 1 * p + base_col + 1
        
        # W indices for C block
        # C[0, base_col] -> W[base_col*n + 0]
        # C[0, base_col+1] -> W[base_col*n + 1]
        # C[1, base_col] -> W[base_col*n + 2]
        # C[1, base_col+1] -> W[base_col*n + 3]
        
        w_c00 = base_col * n + 0
        w_c01 = base_col * n + 1
        w_c10 = base_col * n + 2
        w_c11 = base_col * n + 3
        
        m_base = b * 7
        
        # M1 = (a+d)(e+h)
        U[0][m_base] += 1
        U[3][m_base] += 1
        V[v_e][m_base] += 1
        V[v_h][m_base] += 1
        W[w_c00][m_base] += 1
        W[w_c11][m_base] += 1
        
        # M2 = (c+d)e
        U[2][m_base+1] += 1
        U[3][m_base+1] += 1
        V[v_e][m_base+1] += 1
        W[w_c10][m_base+1] += 1
        W[w_c11][m_base+1] -= 1
        
        # M3 = a(f-g)
        U[0][m_base+2] += 1
        V[v_f][m_base+2] += 1
        V[v_g][m_base+2] -= 1
        W[w_c01][m_base+2] += 1
        W[w_c11][m_base+2] += 1
        
        # M4 = d(g-e)
        U[3][m_base+3] += 1
        V[v_g][m_base+3] += 1
        V[v_e][m_base+3] -= 1
        W[w_c00][m_base+3] += 1
        W[w_c10][m_base+3] += 1
        
        # M5 = (a+b)h
        U[0][m_base+4] += 1
        U[1][m_base+4] += 1
        V[v_h][m_base+4] += 1
        W[w_c00][m_base+4] -= 1
        W[w_c01][m_base+4] += 1
        
        # M6 = (c-a)(e+f)
        U[2][m_base+5] += 1
        U[0][m_base+5] -= 1
        V[v_e][m_base+5] += 1
        V[v_f][m_base+5] += 1
        W[w_c11][m_base+5] += 1
        
        # M7 = (b-d)(g+h)
        U[1][m_base+6] += 1
        U[3][m_base+6] -= 1
        V[v_g][m_base+6] += 1
        V[v_h][m_base+6] += 1
        W[w_c00][m_base+6] += 1

    # Process last block (2x1) at col 10
    col = 10
    v_b0 = 0 * p + col
    v_b1 = 1 * p + col
    
    w_c0 = col * n + 0
    w_c1 = col * n + 1
    
    m_base = 35
    
    # M8 = a * b0
    U[0][m_base] += 1
    V[v_b0][m_base] += 1
    W[w_c0][m_base] += 1
    
    # M9 = b * b1
    U[1][m_base+1] += 1
    V[v_b1][m_base+1] += 1
    W[w_c0][m_base+1] += 1
    
    # M10 = c * b0
    U[2][m_base+2] += 1
    V[v_b0][m_base+2] += 1
    W[w_c1][m_base+2] += 1
    
    # M11 = d * b1
    U[3][m_base+3] += 1
    V[v_b1][m_base+3] += 1
    W[w_c1][m_base+3] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
