"""Rank-55 tensor decomposition for 2x3 @ 3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Dimensions: A is 2x3, B is 3x10, C is 2x10
    n, m, p = 2, 3, 10
    
    # Rank-55 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x1x2 blocks
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    for l in range(5):
        if time.time() > deadline:
            break
        k0 = 2 * l
        k1 = 2 * l + 1
        
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        b0k0 = k0
        b0k1 = k1
        b1k0 = 10 + k0
        b1k1 = 10 + k1
        
        c0k0 = 2 * k0
        c0k1 = 2 * k1
        c1k0 = 2 * k0 + 1
        c1k1 = 2 * k1 + 1
        
        # M1
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
        
        # Schoolbook for j=2
        for i in range(2):
            for k in (k0, k1):
                u_idx = i * 3 + 2
                v_idx = 20 + k
                w_idx = 2 * k + i
                U_mat[u_idx][idx] += 1
                V_mat[v_idx][idx] += 1
                W_mat[w_idx][idx] += 1
                idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
