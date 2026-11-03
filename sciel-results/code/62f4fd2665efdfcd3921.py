"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication using Strassen (rank 7).
    
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    A[i,j]: U indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d -> flat: 0,1,2,3
    B[j,k]: V indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h -> flat: 0,1,2,3
    C[i,k]: W indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,2,1,3 (k*2+i)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a+d)(e+h) -> C[0,0] += M1, C[1,1] += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e -> C[1,0] += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C[1,0] -> k=0,i=1 -> flat 0*2+1=1
    
    # M3 = a(f-h) -> C[0,1] += M3, C[1,0] += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[1][2] = 1  # C[0,1]->flat 2, C[1,0]->flat 1
    
    # M4 = d(g-e) -> C[0,0] += M4, C[1,0] += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h -> C[0,1] -= M5, C[1,1] -= M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = -1; W[3][4] = -1
    
    # M6 = (c-a)(e+f) -> C[1,1] += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h) -> C[0,0] += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(n_rows, m_mid, p_cols):
    """Return U, V, W for n_rows x m_mid x p_cols schoolbook decomposition."""
    rank = n_rows * m_mid * p_cols
    U = [[0]*rank for _ in range(n_rows * m_mid)]
    V = [[0]*rank for _ in range(m_mid * p_cols)]
    W = [[0]*rank for _ in range(p_cols * n_rows)]
    col = 0
    for i in range(n_rows):
        for j in range(m_mid):
            for k in range(p_cols):
                U[i * m_mid + j][col] = 1
                V[j * p_cols + k][col] = 1
                W[k * n_rows + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n_total, m_total, p_total):
    """Combine multiple decompositions covering disjoint index ranges."""
    total_rank = sum(len(U[0]) if U else 0 for U, V, W, _, _, _ in decomps)
    
    U_full = [[0] * total_rank for _ in range(n_total * m_total)]
    V_full = [[0] * total_rank for _ in range(m_total * p_total)]
    W_full = [[0] * total_rank for _ in range(p_total * n_total)]
    
    col_offset = 0
    for U, V, W, (r_start, r_end), (m_start, m_end), (c_start, c_end) in decomps:
        n_r = r_end - r_start
        m_m = m_end - m_start
        p_c = c_end - c_start
        block_rank = len(U[0]) if U else 0
        
        for i_local in range(n_r):
            i_global = r_start + i_local
            for j_local in range(m_m):
                j_global = m_start + j_local
                for c in range(block_rank):
                    U_full[i_global * m_total + j_global][col_offset + c] = U[i_local * m_m + j_local][c]
        
        for j_local in range(m_m):
            j_global = m_start + j_local
            for k_local in range(p_c):
                k_global = c_start + k_local
                for c in range(block_rank):
                    V_full[j_global * p_total + k_global][col_offset + c] = V[j_local * p_c + k_local][c]
        
        for k_local in range(p_c):
            k_global = c_start + k_local
            for i_local in range(n_r):
                i_global = r_start + i_local
                for c in range(block_rank):
                    W_full[k_global * n_total + i_global][col_offset + c] = W[k_local * n_r + i_local][c]
        
        col_offset += block_rank
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    decomps = []
    
    # 2x2 x 2x2 blocks (Strassen, rank 7 each)
    U_s, V_s, W_s = _strassen_2x2x2()
    decomps.append((U_s, V_s, W_s, (0, 2), (0, 2), (0, 2)))
    decomps.append((U_s, V_s, W_s, (0, 2), (2, 4), (0, 2)))
    decomps.append((U_s, V_s, W_s, (0, 2), (0, 2), (2, 4)))
    decomps.append((U_s, V_s, W_s, (0, 2), (2, 4), (2, 4)))
    
    # 2x2 x 2x1 blocks (schoolbook, rank 4 each)
    U_sb, V_sb, W_sb = _schoolbook_block(2, 2, 1)
    decomps.append((U_sb, V_sb, W_sb, (0, 2), (0, 2), (4, 5)))
    decomps.append((U_sb, V_sb, W_sb, (0, 2), (2, 4), (4, 5)))
    
    # 1x2 x 2x2 blocks (schoolbook, rank 4 each)
    U_sb1, V_sb1, W_sb1 = _schoolbook_block(1, 2, 2)
    decomps.append((U_sb1, V_sb1, W_sb1, (2, 3), (0, 2), (0, 2)))
    decomps.append((U_sb1, V_sb1, W_sb1, (2, 3), (2, 4), (0, 2)))
    decomps.append((U_sb1, V_sb1, W_sb1, (2, 3), (0, 2), (2, 4)))
    decomps.append((U_sb1, V_sb1, W_sb1, (2, 3), (2, 4), (2, 4)))
    
    # 1x2 x 2x1 blocks (schoolbook, rank 2 each)
    U_sb2, V_sb2, W_sb2 = _schoolbook_block(1, 2, 1)
    decomps.append((U_sb2, V_sb2, W_sb2, (2, 3), (0, 2), (4, 5)))
    decomps.append((U_sb2, V_sb2, W_sb2, (2, 3), (2, 4), (4, 5)))
    
    U_full, V_full, W_full = _combine_decompositions(decomps, n, m, p)
    
    # Verify
    if time.time() < deadline:
        _verify(U_full, V_full, W_full, n, m, p)
    
    return {"U": U_full, "V": V_full, "W": W_full}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k},val={val}")

# EVOLVE-BLOCK-END
