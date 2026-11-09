"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Split m = 5 into [2, 2, 1]
    m_parts = [2, 2, 1]
    # Split n = 4 into [2, 2]
    n_parts = [2, 2]
    # Split p = 5 into [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Calculate total rank
    total_rank = 0
    for mp in m_parts:
        for np_ in n_parts:
            for pp in p_parts:
                if np_ == 2 and mp == 2 and pp == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += np_ * mp * pp  # Standard
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    j_offset = 0
    for mp in m_parts:
        i_offset = 0
        for np_ in n_parts:
            k_offset = 0
            for pp in p_parts:
                if np_ == 2 and mp == 2 and pp == 2:
                    col = add_strassen(2, 2, 2, i_offset, j_offset, k_offset, n, m, p, U, V, W, col)
                else:
                    col = add_standard(np_, mp, pp, i_offset, j_offset, k_offset, n, m, p, U, V, W, col)
                k_offset += pp
            i_offset += np_
        j_offset += mp
    
    return {"U": U, "V": V, "W": W}

def add_standard(n_sub, m_sub, p_sub, i_off, j_off, k_off, n, m, p, U, V, W, col):
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                u_idx = (i_off + i) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return col

def add_strassen(n_sub, m_sub, p_sub, i_off, j_off, k_off, n, m, p, U, V, W, col):
    # Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    U[(i_off+0)*m + (j_off+0)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+0)*p + (k_off+0)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[(i_off+1)*m + (j_off+0)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+0)*p + (k_off+0)][col] = 1
    W[(k_off+0)*n + (i_off+1)][col] = 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[(i_off+0)*m + (j_off+0)][col] = 1
    V[(j_off+0)*p + (k_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = -1
    W[(k_off+1)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+0)][col] = 1
    V[(j_off+0)*p + (k_off+0)][col] = -1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+0)*n + (i_off+1)][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[(i_off+0)*m + (j_off+0)][col] = 1
    U[(i_off+0)*m + (j_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[(k_off+0)*n + (i_off+0)][col] = -1
    W[(k_off+1)*n + (i_off+0)][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[(i_off+1)*m + (j_off+0)][col] = 1
    U[(i_off+0)*m + (j_off+0)][col] = -1
    V[(j_off+0)*p + (k_off+0)][col] = 1
    V[(j_off+0)*p + (k_off+1)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[(i_off+0)*m + (j_off+1)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = -1
    V[(j_off+1)*p + (k_off+0)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[(k_off+0)*n + (i_off+0)][col] = 1
    W[(k_off+1)*n + (i_off+0)][col] = -1
    col += 1
    
    return col
# EVOLVE-BLOCK-END
