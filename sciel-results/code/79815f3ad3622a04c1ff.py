"""Rewritten program exploiting block-diagonal decomposition for M_{2,3,10}."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    n, m, p = 2, 3, 10
    rank = 55
    
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Helper to add a Strassen block for 2x2 * 2x2
    def add_strassen_block(a00, a01, a10, a11, b0k0, b0k1, b1k0, b1k1, c0k0, c0k1, c1k0, c1k1):
        nonlocal idx
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1

    # Block 1: 2x2 sub-problem (cols 0,1 of A; rows 0,1 of B)
    for l in range(5):
        if time.time() > deadline:
            return {'U': U_mat, 'V': V_mat, 'W': W_mat}

        k0 = 2 * l
        k1 = 2 * l + 1
        
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        b0k0 = 0 * p + k0
        b0k1 = 0 * p + k1
        b1k0 = 1 * p + k0
        b1k1 = 1 * p + k1
        
        c0k0 = k0 * n + 0
        c0k1 = k1 * n + 0
        c1k0 = k0 * n + 1
        c1k1 = k1 * n + 1
        
        add_strassen_block(a00, a01, a10, a11, b0k0, b0k1, b1k0, b1k1, c0k0, c0k1, c1k0, c1k1)
        
    # Block 2: Schoolbook for col 2 of A and row 2 of B
    for i in range(n):
        for k in range(p):
            if time.time() > deadline:
                return {'U': U_mat, 'V': V_mat, 'W': W_mat}

            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
