import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    s_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]
    
    col = 0
    for r in range(7):
        for s in range(7):
            if time.time() > deadline:
                break
            u_r, v_r, w_r = s_terms[r]
            u_s, v_s, w_s = s_terms[s]
            
            u_4x4 = np.zeros(16)
            v_4x4 = np.zeros(16)
            w_4x4 = np.zeros(16)
            
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    ci, cj = i % 2, j % 2
                    u_4x4[i*4+j] = u_r[bi*2+bj] * u_s[ci*2+cj]
                    v_4x4[i*4+j] = v_r[bi*2+bj] * v_s[ci*2+cj]
                    w_4x4[i*4+j] = w_r[bi*2+bj] * w_s[ci*2+cj]
                    
            U[:, col] = u_4x4
            V[:, col] = v_4x4
            W[:, col] = w_4x4
            col += 1
            
    W_new = np.zeros_like(W)
    for i in range(4):
        for k in range(4):
            W_new[k*4+i] = W[i*4+k]
    W = W_new
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
