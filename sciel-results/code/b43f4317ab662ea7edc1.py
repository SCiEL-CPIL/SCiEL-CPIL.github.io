"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strassen blocks: (i_range, j_range, k_range)
    strassen_blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
    ]
    
    terms = []  # Each term is (U_vec, V_vec, W_vec)
    
    # Add Strassen terms
    for (i_range, j_range, k_range) in strassen_blocks:
        i0, i1 = i_range[0], i_range[1]
        j0, j1 = j_range[0], j_range[1]
        k0, k1 = k_range[0], k_range[1]
        
        # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        u1 = np.zeros(n*m); u1[i0*m+j0] = 1; u1[i1*m+j1] = 1
        v1 = np.zeros(m*p); v1[j0*p+k0] = 1; v1[j1*p+k1] = 1
        w1 = np.zeros(p*n); w1[k0*n+i0] = 1; w1[k1*n+i1] = 1
        terms.append((u1, v1, w1))
        
        # m2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        u2 = np.zeros(n*m); u2[i1*m+j0] = 1; u2[i1*m+j1] = 1
        v2 = np.zeros(m*p); v2[j0*p+k0] = 1
        w2 = np.zeros(p*n); w2[k0*n+i1] = 1
        terms.append((u2, v2, w2))
        
        # m3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        u3 = np.zeros(n*m); u3[i0*m+j0] = 1
        v3 = np.zeros(m*p); v3[j0*p+k1] = 1; v3[j1*p+k1] = -1
        w3 = np.zeros(p*n); w3[k1*n+i0] = 1
        terms.append((u3, v3, w3))
        
        # m4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        u4 = np.zeros(n*m); u4[i1*m+j1] = 1
        v4 = np.zeros(m*p); v4[j1*p+k0] = 1; v4[j0*p+k0] = -1
        w4 = np.zeros(p*n); w4[k0*n+i1] = 1; w4[k0*n+i0] = 1
        terms.append((u4, v4, w4))
        
        # m5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        u5 = np.zeros(n*m); u5[i0*m+j0] = 1; u5[i0*m+j1] = 1
        v5 = np.zeros(m*p); v5[j1*p+k1] = 1
        w5 = np.zeros(p*n); w5[k1*n+i1] = -1; w5[k1*n+i0] = -1
        terms.append((u5, v5, w5))
        
        # m6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        u6 = np.zeros(n*m); u6[i1*m+j0] = 1; u6[i0*m+j0] = -1
        v6 = np.zeros(m*p); v6[j0*p+k0] = 1; v6[j0*p+k1] = 1
        w6 = np.zeros(p*n); w6[k1*n+i1] = 1
        terms.append((u6, v6, w6))
        
        # m7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        u7 = np.zeros(n*m); u7[i0*m+j1] = 1; u7[i1*m+j1] = -1
        v7 = np.zeros(m*p); v7[j1*p+k0] = 1; v7[j1*p+k1] = 1
        w7 = np.zeros(p*n); w7[k0*n+i0] = 1
        terms.append((u7, v7, w7))
    
    # Add schoolbook terms for remaining (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                for (i_range, j_range, k_range) in strassen_blocks:
                    if i in i_range and j in j_range and k in k_range:
                        covered = True
                        break
                if not covered:
                    u = np.zeros(n*m); u[i*m+j] = 1
                    v = np.zeros(m*p); v[j*p+k] = 1
                    w = np.zeros(p*n); w[k*n+i] = 1
                    terms.append((u, v, w))
    
    R = len(terms)
    U = np.column_stack([t[0] for t in terms])
    V = np.column_stack([t[1] for t in terms])
    W = np.column_stack([t[2] for t in terms])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
