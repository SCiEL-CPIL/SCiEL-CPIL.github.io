<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][column] = 1
                V[idx_v][column] = 1
                W[idx_w][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np

def get_strassen_block(n=2, m=2, p=2):
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1
    U[0*m+0][0] = 1; U[1*m+1][0] = 1
    V[0*p+0][0] = 1; V[1*p+1][0] = 1
    W[0*n+0][0] = 1; W[1*n+1][0] = 1
    
    # M2
    U[1*m+0][1] = 1; U[1*m+1][1] = 1
    V[0*p+0][1] = 1
    W[1*n+0][1] = 1; W[1*n+1][1] = 1
    
    # M3
    U[0*m+0][2] = 1
    V[0*p+1][2] = 1; V[1*p+1][2] = -1
    W[0*n+1][2] = 1; W[1*n+1][2] = 1
    
    # M4
    U[1*m+1][3] = 1
    V[1*p+0][3] = 1; V[0*p+0][3] = -1
    W[0*n+0][3] = 1; W[1*n+0][3] = 1
    
    # M5
    U[0*m+0][4] = 1; U[0*m+1][4] = 1
    V[1*p+1][4] = 1
    W[0*n+0][4] = 1; W[0*n+1][4] = 1
    
    # M6
    U[1*m+0][5] = 1; U[0*m+0][5] = -1
    V[0*p+0][5] = 1; V[0*p+1][5] = 1
    W[1*n+1][5] = 1
    
    # M7
    U[0*m+1][6] = 1; U[1*m+1][6] = -1
    V[1*p+0][6] = 1; V[1*p+1][6] = 1
    W[0*n+0][6] = 1; W[1*n+1][6] = 1
    
    return U, V, W

def get_schoolbook_block(n, m, p):
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
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
    n, m, p = DIMENSIONS
    total_rank = 99
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    def add_block(n_s, m_s, p_s, method, i_off, j_off, k_off):
        nonlocal col_offset
        if method == 'strassen':
            u, v, w = get_strassen_block(n_s, m_s, p_s)
            rank_s = 7
        else:
            u, v, w = get_schoolbook_block(n_s, m_s, p_s)
            rank_s = n_s * m_s * p_s
            
        for i in range(n_s):
            for j in range(m_s):
                U[(i+i_off)*m + (j+j_off), col_offset:col_offset+rank_s] = u[i*m_s+j, :]
                
        for j in range(m_s):
            for k in range(p_s):
                V[(j+j_off)*p + (k+k_off), col_offset:col_offset+rank_s] = v[j*p_s+k, :]
                
        for k in range(p_s):
            for i in range(n_s):
                W[(k+k_off)*n + (i+i_off), col_offset:col_offset+rank_s] = w[k*n_s+i, :]
                
        col_offset += rank_s

    add_block(2, 2, 2, 'strassen', 0, 0, 0)
    add_block(2, 2, 5, 'schoolbook', 0, 0, 2)
    add_block(1, 2, 2, 'schoolbook', 2, 0, 0)
    add_block(1, 2, 5, 'schoolbook', 2, 0, 2)
    
    add_block(2, 2, 2, 'strassen', 0, 2, 0)
    add_block(2, 2, 5, 'schoolbook', 0, 2, 2)
    add_block(2, 1, 5, 'schoolbook', 0, 4, 2)
    add_block(1, 2, 2, 'schoolbook', 2, 2, 0)
    add_block(1, 1, 2, 'schoolbook', 2, 4, 0)
    add_block(1, 2, 5, 'schoolbook', 2, 2, 2)
    add_block(1, 1, 5, 'schoolbook', 2, 4, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
