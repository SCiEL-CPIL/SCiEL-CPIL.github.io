import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # We will decompose M_{3,4,7} into blocks:
    # Split m=4 into 3+1
    # Split p=7 into 3+3+1
    # Blocks:
    # 1. M_{3,3,3} (i:0-2, j:0-2, k:0-2)
    # 2. M_{3,3,3} (i:0-2, j:0-2, k:3-5)
    # 3. M_{3,3,1} (i:0-2, j:0-2, k:6)
    # 4. M_{3,1,3} (i:0-2, j:3, k:0-2)
    # 5. M_{3,1,3} (i:0-2, j:3, k:3-5)
    # 6. M_{3,1,1} (i:0-2, j:3, k:6)
    
    # Ranks for schoolbook:
    # 1. 27
    # 2. 27
    # 3. 9
    # 4. 3
    # 5. 3
    # 6. 3
    # Total: 72
    
    rank = 27 + 27 + 9 + 3 + 3 + 3
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_schoolbook(i_off, j_off, k_off, l_n, l_m, l_p, r_off):
        for li in range(l_n):
            for lj in range(l_m):
                for lk in range(l_p):
                    r = r_off + (li * l_m + lj) * l_p + lk
                    # U index: (i_off + li) * m + (j_off + lj)
                    u_idx = (i_off + li) * m + (j_off + lj)
                    # V index: (j_off + lj) * p + (k_off + lk)
                    v_idx = (j_off + lj) * p + (k_off + lk)
                    # W index: (k_off + lk) * n + (i_off + li)
                    w_idx = (k_off + lk) * n + (i_off + li)
                    
                    U[u_idx][r] = 1
                    V[v_idx][r] = 1
                    W[w_idx][r] = 1
                    if time.time() > deadline:
                        raise TimeoutError("Time limit exceeded")

    # Block 1: M_{3,3,3}
    add_schoolbook(0, 0, 0, 3, 3, 3, 0)
    # Block 2: M_{3,3,3}
    add_schoolbook(0, 0, 3, 3, 3, 3, 27)
    # Block 3: M_{3,3,1}
    add_schoolbook(0, 0, 6, 3, 3, 1, 54)
    # Block 4: M_{3,1,3}
    add_schoolbook(0, 3, 0, 3, 1, 3, 63)
    # Block 5: M_{3,1,3}
    add_schoolbook(0, 3, 3, 3, 1, 3, 66)
    # Block 6: M_{3,1,1}
    add_schoolbook(0, 3, 6, 3, 1, 1, 69)
    
    return {"U": U, "V": V, "W": W}
