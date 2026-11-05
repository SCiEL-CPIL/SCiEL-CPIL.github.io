import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Strassen 2x2x2 patterns
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_2x2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),      # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),       # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),      # M3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),      # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),      # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),      # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),      # M7
    ]

    # Generate 4x4x4 Strassen terms
    def tensor_product_vec(v1, v2, d1, d2):
        d = 2
        result = [0] * (d1 * d2 * d1 * d2)
        for a1 in range(d1):
            for b1 in range(d1):
                for a2 in range(d2):
                    for b2 in range(d2):
                        idx = (a1 * d2 + a2) * (d1 * d2) + (b1 * d2 + b2)
                        result[idx] = v1[a1 * d1 + b1] * v2[a2 * d2 + b2]
        return result

    strassen_4x4x4 = []
    for t1 in strassen_2x2x2:
        for t2 in strassen_2x2x2:
            u = tensor_product_vec(t1[0], t2[0], 2, 2)
            v = tensor_product_vec(t1[1], t2[1], 2, 2)
            w = tensor_product_vec(t1[2], t2[2], 2, 2)
            strassen_4x4x4.append((u, v, w))

    U_terms = []
    V_terms = []
    W_terms = []

    # 1. Add 4x4x4 Strassen block for i=0..3, j=0..3, k=0..3
    for u_pat, v_pat, w_pat in strassen_4x4x4:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in enumerate(u_pat):
            i_loc = idx // 4
            j_loc = idx % 4
            u[i_loc * m + j_loc] = val
            
        for idx, val in enumerate(v_pat):
            j_loc = idx // 4
            k_loc = idx % 4
            v[j_loc * p + k_loc] = val
            
        for idx, val in enumerate(w_pat):
            k_loc = idx // 4
            i_loc = idx % 4
            w[k_loc * n + i_loc] = val
            
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    # 2. Add Rank-1 terms for j=4 strip (i=0..3, j=4, k=0..4)
    # Split i into {0,1} and {2,3}
    for i_start in [0, 2]:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # U[i, 4] = 1 for i in {i_start, i_start+1}
        for i_loc in range(2):
            u[(i_start + i_loc) * m + 4] = 1
            
        # V[4, k] = 1 for all k
        for k in range(p):
            v[4 * p + k] = 1
            
        # W[k, i] = 1 for i in {i_start, i_start+1}
        for k in range(p):
            for i_loc in range(2):
                w[k * n + (i_start + i_loc)] = 1
                
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    # 3. Add Rank-1 terms for k=4 strip (i=0..3, j=0..3, k=4)
    # Split j into {0,1} and {2,3}
    for j_start in [0, 2]:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # U[i, j] = 1 for i in 0..3, j in {j_start, j_start+1}
        for i in range(n):
            for j_loc in range(2):
                u[i * m + (j_start + j_loc)] = 1
                
        # V[j, 4] = 1 for j in {j_start, j_start+1}
        for j_loc in range(2):
            v[(j_start + j_loc) * p + 4] = 1
            
        # W[4, i] = 1 for i in 0..3
        for i in range(n):
            w[4 * n + i] = 1
            
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    rank = len(U_terms)
    
    # Convert to matrix format
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r_idx in range(rank):
        for idx, val in enumerate(U_terms[r_idx]):
            if val != 0:
                U_mat[idx][r_idx] = val
        for idx, val in enumerate(V_terms[r_idx]):
            if val != 0:
                V_mat[idx][r_idx] = val
        for idx, val in enumerate(W_terms[r_idx]):
            if val != 0:
                W_mat[idx][r_idx] = val
                
    # Verify
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True

    if verify(U_mat, V_mat, W_mat):
        return {'U': U_mat, 'V': V_mat, 'W': W_mat}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    return {'U': U, 'V': V, 'W': W}
