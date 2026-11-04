"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U,V,W for 2x2x2 Strassen decomposition. Rank 7."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # t=0: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # t=1: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # t=2: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # t=3: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # t=4: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # t=5: M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # t=6: M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m times m x p. Rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Block decomposition of 5x5x5 using (2,2,1) split with Strassen for 2x2x2 blocks.
    Expected rank: 117
    """
    n, m, p = DIMENSIONS
    
    n_splits = [2, 2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    sub_decomps = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                ni = n_splits[i]
                mk = m_splits[k]
                pj = p_splits[j]
                
                if ni == 2 and mk == 2 and pj == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2()
                else:
                    U_sub, V_sub, W_sub = _schoolbook_nm_p(ni, mk, pj)
                
                u_row_off = n_offsets[i] * m + m_offsets[k]
                v_row_off = m_offsets[k] * p + p_offsets[j]
                w_row_off = p_offsets[j] * n + n_offsets[i]
                
                sub_decomps.append((U_sub, V_sub, W_sub, u_row_off, v_row_off, w_row_off))
    
    total_rank = sum(len(U[0]) for U, V, W, _, _, _ in sub_decomps)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_sub, V_sub, W_sub, u_off, v_off, w_off in sub_decomps:
        rank_sub = len(U_sub[0])
        for r in range(len(U_sub)):
            for c in range(rank_sub):
                U[u_off + r][col_offset + c] = U_sub[r][c]
                V[v_off + r][col_offset + c] = V_sub[r][c]
                W[w_off + r][col_offset + c] = W_sub[r][c]
        col_offset += rank_sub
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
