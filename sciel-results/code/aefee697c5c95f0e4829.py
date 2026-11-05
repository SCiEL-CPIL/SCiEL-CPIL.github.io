import numpy as np

def _schoolbook_factors(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_factors():
    rank = 7
    U = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0]
    ]
    V = [
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, -1, 1]
    ]
    W = [
        [1, 0, 0, -1, 1, 0, 0],
        [0, 1, 0, 0, -1, 0, 1],
        [0, 0, 1, 1, 0, -1, 0],
        [1, 0, -1, 0, 0, 1, 0]
    ]
    return U, V, W

def construct():
    n, m, p = 5, 5, 5
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    total_rank = 0
    # Pre-calculate total rank
    for I in range(3):
        for J in range(3):
            for K in range(3):
                n_I = sizes[I]
                m_K = sizes[K]
                p_J = sizes[J]
                if n_I == 2 and m_K == 2 and p_J == 2:
                    total_rank += 7
                else:
                    total_rank += n_I * m_K * p_J
                    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                n_I = sizes[I]
                m_K = sizes[K]
                p_J = sizes[J]
                
                if n_I == 2 and m_K == 2 and p_J == 2:
                    U_L, V_L, W_L = _strassen_factors()
                    r_L = 7
                else:
                    U_L, V_L, W_L = _schoolbook_factors(n_I, m_K, p_J)
                    r_L = n_I * m_K * p_J
                    
                for r in range(r_L):
                    for i in range(n_I):
                        for j in range(m_K):
                            u_row = i * m + j
                            u_col = rank_offset + r
                            U[u_row][u_col] = U_L[i * m_K + j][r]
                            
                            for k in range(p_J):
                                v_row = j * p + k
                                v_col = rank_offset + r
                                V[v_row][v_col] = V_L[j * p_J + k][r]
                                
                                w_row = k * n + i
                                w_col = rank_offset + r
                                W[w_row][w_col] = W_L[k * n_I + i][r]
                                
                rank_offset += r_L
                
    return {"U": U, "V": V, "W": W}
