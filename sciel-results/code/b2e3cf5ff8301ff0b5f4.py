import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U_mat = np.zeros((n * m, rank), dtype=int)
    V_mat = np.zeros((m * p, rank), dtype=int)
    W_mat = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    
    # Term 1: Strassen blocks for 5 pairs of columns in B
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices (2x3 matrix, flat index i*3+j)
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        # B indices (3x10 matrix, flat index j*10+k)
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # W indices (10x2 matrix? No, C is 2x10. W is p*n = 20. Index k*n+i)
        # C[i,k] corresponds to W[k*n + i]
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00, idx] += 1
        U_mat[a11, idx] += 1
        V_mat[b0k0, idx] += 1
        V_mat[b1k1, idx] += 1
        W_mat[c0k0, idx] += 1
        W_mat[c1k1, idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10, idx] += 1
        U_mat[a11, idx] += 1
        V_mat[b0k0, idx] += 1
        W_mat[c1k0, idx] += 1
        W_mat[c1k1, idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00, idx] += 1
        V_mat[b0k1, idx] += 1
        V_mat[b1k1, idx] -= 1
        W_mat[c0k1, idx] += 1
        W_mat[c1k1, idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11, idx] += 1
        V_mat[b1k0, idx] += 1
        V_mat[b0k0, idx] -= 1
        W_mat[c0k0, idx] += 1
        W_mat[c1k0, idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00, idx] += 1
        U_mat[a01, idx] += 1
        V_mat[b1k1, idx] += 1
        W_mat[c0k0, idx] -= 1
        W_mat[c0k1, idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10, idx] += 1
        U_mat[a00, idx] -= 1
        V_mat[b0k0, idx] += 1
        V_mat[b0k1, idx] += 1
        W_mat[c1k1, idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01, idx] += 1
        U_mat[a11, idx] -= 1
        V_mat[b1k0, idx] += 1
        V_mat[b1k1, idx] += 1
        W_mat[c0k0, idx] += 1
        idx += 1
        
    # Term 2: Schoolbook for j=2 (A[:, 2] * B[2, :])
    # A[:, 2] is vector of size 2. B[2, :] is vector of size 10.
    # Product is 2x10 matrix.
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx, idx] += 1
            V_mat[v_idx, idx] += 1
            W_mat[w_idx, idx] += 1
            idx += 1
            
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
