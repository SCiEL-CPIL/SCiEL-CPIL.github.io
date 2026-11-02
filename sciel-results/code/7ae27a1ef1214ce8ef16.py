import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Rank is 28
    rank = 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    # 4 blocks of 2x2 * 2x2 -> 2x2
    # A is 2x4, B is 4x4, C is 2x4
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        (0, 0, 0, 0, 0), # a_r, a_c, b_r, b_c, c_c
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    current_rank = 0
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        # Local indices
        u11 = u_idx(a_r, a_c)
        u12 = u_idx(a_r, a_c + 1)
        u21 = u_idx(a_r + 1, a_c)
        u22 = u_idx(a_r + 1, a_c + 1)
        
        v11 = v_idx(b_r, b_c)
        v12 = v_idx(b_r, b_c + 1)
        v21 = v_idx(b_r + 1, b_c)
        v22 = v_idx(b_r + 1, b_c + 1)
        
        w11 = w_idx(0, c_c)
        w12 = w_idx(0, c_c + 1)
        w21 = w_idx(1, c_c)
        w22 = w_idx(1, c_c + 1)
        
        # Strassen 7 terms
        
        # M1 = (u11 + u22) * (v11 + v22) -> w11 += M1, w22 += M1
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M2 = (u21 + u22) * v11 -> w21 += M2, w22 -= M2
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # M3 = u11 * (v12 - v22) -> w12 += M3, w22 += M3
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M4 = u22 * (v21 - v11) -> w11 += M4, w21 += M4
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # M5 = (u11 + u12) * v22 -> w11 -= M5, w12 += M5
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # M6 = (u21 - u11) * (v11 + v12) -> w22 += M6
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M7 = (u12 - u22) * (v21 + v22) -> w11 += M7
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
        
    if current_rank != rank:
        raise ValueError("Rank mismatch")
        
    return {"U": U, "V": V, "W": W}
