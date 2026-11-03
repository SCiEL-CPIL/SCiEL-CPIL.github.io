import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, keep 10s margin
    
    n, m, p = 2, 4, 4
    
    # We will use the known Strassen-based decomposition for 2x4x4 matrix multiplication
    # which achieves rank 28 (4 blocks of rank-7 Strassen 2x2x2)
    
    TARGET_RANK = 28
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i
    
    # The 4 blocks of the 2x4x4 multiplication, each being a 2x2x2 Strassen block
    # Block (a_r, a_c, b_r, b_c, c_c) means:
    #   A rows [a_r, a_r+1], cols [a_c, a_c+1]
    #   B rows [b_r, b_r+1], cols [b_c, b_c+1]
    #   C rows [c_c, c_c+1] (in the output of row a_r and a_r+1)
    blocks = [
        (0, 0, 0, 0, 0),   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0),   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 2),   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 2),   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        if time.time() > deadline:
            break
            
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        w11 = get_w_idx(0, c_c)
        w12 = get_w_idx(0, c_c + 1)
        w21 = get_w_idx(1, c_c)
        w22 = get_w_idx(1, c_c + 1)
        
        # Strassen 2x2x2 decomposition (7 multiplications)
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 1: M1 = (A11 + A22) * (B11 + B22)
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 2: M2 = (A21 + A22) * B11
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # Term 3: M3 = A11 * (B12 - B22)
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 4: M4 = A22 * (B21 - B11)
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # Term 5: M5 = (A11 + A12) * B22
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # Term 6: M6 = (A21 - A11) * (B11 + B12)
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 7: M7 = (A12 - A22) * (B21 + B22)
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
    
    if current_rank != TARGET_RANK:
        # If we broke early, we have an invalid result.
        pass
        
    return {"U": U, "V": V, "W": W}
