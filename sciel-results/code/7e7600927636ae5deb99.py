import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Strassen decomposition for 2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # U row: i*2 + j
    # V row: j*2 + k
    # W row: k*2 + i
    
    U_strassen = [[0]*7 for _ in range(4)]
    V_strassen = [[0]*7 for _ in range(4)]
    W_strassen = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (a00+a11)(b00+b11) -> c00 + c11
    U_strassen[0][0] = 1; U_strassen[3][0] = 1
    V_strassen[0][0] = 1; V_strassen[3][0] = 1
    W_strassen[0][0] = 1; W_strassen[3][0] = 1
    
    # Term 1: M2 = (a10+a11)b00 -> c10 - c11
    U_strassen[2][1] = 1; U_strassen[3][1] = 1
    V_strassen[0][1] = 1
    W_strassen[2][1] = 1; W_strassen[3][1] = -1
    
    # Term 2: M3 = a00(b01-b11) -> c01 + c11
    U_strassen[0][2] = 1
    V_strassen[1][2] = 1; V_strassen[3][2] = -1
    W_strassen[1][2] = 1; W_strassen[3][2] = 1
    
    # Term 3: M4 = a11(b10-b00) -> c00 + c10
    U_strassen[3][3] = 1
    V_strassen[2][3] = 1; V_strassen[0][3] = -1
    W_strassen[0][3] = 1; W_strassen[2][3] = 1
    
    # Term 4: M5 = (a00+a01)b11 -> -c00 + c01
    U_strassen[0][4] = 1; U_strassen[1][4] = 1
    V_strassen[3][4] = 1
    W_strassen[0][4] = -1; W_strassen[1][4] = 1
    
    # Term 5: M6 = (a10-a00)(b00+b01) -> c11
    U_strassen[2][5] = 1; U_strassen[0][5] = -1
    V_strassen[0][5] = 1; V_strassen[1][5] = 1
    W_strassen[3][5] = 1
    
    # Term 6: M7 = (a01-a11)(b10+b11) -> c00
    U_strassen[1][6] = 1; U_strassen[3][6] = -1
    V_strassen[2][6] = 1; V_strassen[3][6] = 1
    W_strassen[0][6] = 1

    # Construct full matrices
    # n=2, m=2, p=16
    # U size: 4 x 56
    # V size: 32 x 56
    # W size: 32 x 56
    
    rank = 7 * (p // 2) # 7 * 8 = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for g in range(p // 2):
        base_r = g * 7
        # Copy U
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for r_loc in range(7):
                    U[u_idx][base_r + r_loc] = U_strassen[u_idx][r_loc]
                    
        # Copy V
        for j in range(m):
            for k_loc in range(2):
                k_glob = 2 * g + k_loc
                v_idx_glob = j * p + k_glob
                v_idx_loc = j * 2 + k_loc
                for r_loc in range(7):
                    V[v_idx_glob][base_r + r_loc] = V_strassen[v_idx_loc][r_loc]
                    
        # Copy W
        for k_loc in range(2):
            k_glob = 2 * g + k_loc
            for i in range(n):
                w_idx_glob = k_glob * n + i
                w_idx_loc = k_loc * n + i
                for r_loc in range(7):
                    W[w_idx_glob][base_r + r_loc] = W_strassen[w_idx_loc][r_loc]
                    
    # Verify correctness quickly (optional, but good for confidence)
    # Check if sum_r U[a,r]V[b,r]W[c,r] matches T[a,b,c]
    # T[a,b,c] = 1 if a=im+j, b=jp+k, c=kn+i
    # We can sample a few points.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
