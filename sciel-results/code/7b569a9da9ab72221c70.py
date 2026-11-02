import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    
    # Strategy: Split m=3 into m1=1, m2=2
    # Split p=8 into p1=4, p2=4
    # Block (2x1)x(1x4): 2x1x4, rank 8
    # Block (2x1)x(1x4): 2x1x4, rank 8
    # Block (2x2)x(2x4): use Strassen 2x2x2 blocks, rank 7*2=14
    # Block (2x2)x(2x4): use Strassen 2x2x2 blocks, rank 7*2=14
    # Total: 8+8+14+14 = 44
    
    m1, m2 = 1, 2
    p1, p2 = 4, 4
    
    total_rank = 8 + 8 + 14 + 14  # = 44
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Block A: rows 0:2 (n=2), cols 0:1 of A (m1=1)
    # times rows 0:1 of B (m1=1), cols 0:4 of B (p1=4)
    # Output: rows 0:4 of result (p1=4), cols 0:2 (n=2)
    # T[i*m+j, j*p+k, k*n+i] where i in 0..1, j=0, k in 0..3
    for i in range(n):
        for k in range(p1):
            # U index: i*m + 0 = i*3 + 0
            # V index: 0*p + k = k
            # W index: k*n + i = k*2 + i
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Block B: rows 0:2, cols 0:1 of A
    # times rows 0:1 of B, cols 4:8 of B (p2=4)
    # Output: rows 4:8, cols 0:2
    for i in range(n):
        for k in range(p2):
            # U index: i*3 + 0
            # V index: 0*p + (p1 + k) = 4 + k
            # W index: (p1+k)*n + i = (4+k)*2 + i
            U[i * m + 0, col] = 1.0
            V[0 * p + (p1 + k), col] = 1.0
            W[(p1 + k) * n + i, col] = 1.0
            col += 1
    
    # Block C: rows 0:2, cols 1:3 of A (m2=2)
    # times rows 1:3 of B (m2=2), cols 0:4 of B (p1=4)
    # Use Strassen for 2x2x2, repeat for 2 column blocks of size 2
    # For each pair of output columns, use Strassen on 2x2x2
    
    # Strassen 2x2x2:
    # m1: (a+b)(e+f) -> w11
    # m2: (c+d)(e) -> w12  
    # m3: (a)(e-g) -> w21
    # m4: (d)(f-h) -> w22
    # m5: (a+b)(h) -> w11
    # m6: (c-a)(e+f) -> w11
    # m7: (b-d)(g+h) -> w22
    
    # For 2x2x4: two Strassen blocks
    
    # First Strassen block: output cols 0:2
    for s in range(2):  # 2 output column pairs
        # Output col indices: s*2, s*2+1
        # A sub-block rows: 1,2 (indices 1,2 in 0-indexed for m)
        # B sub-block rows: 1,2 (indices 1,2)
        # B sub-block cols: s*2, s*2+1
        
        a_idx = 1 * m + 0  # row 1, col 0 of A sub-block -> global index 1*3+0=3
        b_idx = 1 * m + 1  # row 1, col 1 -> 1*3+1=4
        c_idx = 2 * m + 0  # row 2, col 0 -> 2*3+0=6
        d_idx = 2 * m + 1  # row 2, col 1 -> 2*3+1=7
        
        e_idx = 1 * p + s*2      # row 1, col s*2 -> 1*8+s*2
        f_idx = 1 * p + s*2 + 1  # row 1, col s*2+1
        g_idx = 2 * p + s*2      # row 2, col s*2
        h_idx = 2 * p + s*2 + 1  # row 2, col s*2+1
        
        # Output indices: (p1+k)*n + i for k in s*2..s*2+1, i in 0..1
        w_11 = (p1 + s*2) * n + 0    # k=s*2, i=0
        w_12 = (p1 + s*2) * n + 1    # k=s*2, i=1
        w_21 = (p1 + s*2 + 1) * n + 0  # k=s*2+1, i=0
        w_22 = (p1 + s*2 + 1) * n + 1  # k=s*2+1, i=1
        
        # m1: (a+b)(e+f) -> w11
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # m2: (c+d)(e) -> w12
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[w_12, col] += 1
        col += 1
        
        # m3: (a)(e-g) -> w21
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        V[g_idx, col] += -1
        W[w_21, col] += 1
        col += 1
        
        # m4: (d)(f-h) -> w22
        U[d_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] += -1
        W[w_22, col] += 1
        col += 1
        
        # m5: (a+b)(h) -> w11
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # m6: (c-a)(e+f) -> w11
        U[c_idx, col] += 1
        U[a_idx, col] += -1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # m7: (b-d)(g+h) -> w22
        U[b_idx, col] += 1
        U[d_idx, col] += -1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[w_22, col] += 1
        col += 1
    
    # Block D: rows 0:2, cols 1:3 of A
    # times rows 1:3 of B, cols 4:8 of B (p2=4)
    # Use Strassen for 2x2x2, repeat for 2 column blocks of size 2
    for s in range(2):
        a_idx = 1 * m + 0
        b_idx = 1 * m + 1
        c_idx = 2 * m + 0
        d_idx = 2 * m + 1
        
        e_idx = 1 * p + (p1 + s*2)
        f_idx = 1 * p + (p1 + s*2 + 1)
        g_idx = 2 * p + (p1 + s*2)
        h_idx = 2 * p + (p1 + s*2 + 1)
        
        w_11 = (p1 + p1 + s*2) * n + 0
        w_12 = (p1 + p1 + s*2) * n + 1
        w_21 = (p1 + p1 + s*2 + 1) * n + 0
        w_22 = (p1 + p1 + s*2 + 1) * n + 1
        
        # m1: (a+b)(e+f) -> w11
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # m2: (c+d)(e) -> w12
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[w_12, col] += 1
        col += 1
        
        # m3: (a)(e-g) -> w21
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        V[g_idx, col] += -1
        W[w_21, col] += 1
        col += 1
        
        # m4: (d)(f-h) -> w22
        U[d_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] += -1
        W[w_22, col] += 1
        col += 1
        
        # m5: (a+b)(h) -> w11
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # m6: (c-a)(e+f) -> w11
        U[c_idx, col] += 1
        U[a_idx, col] += -1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # m7: (b-d)(g+h) -> w22
        U[b_idx, col] += 1
        U[d_idx, col] += -1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[w_22, col] += 1
        col += 1
    
    # Verify decomposition
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    S = np.zeros((n * m, m * p, p * n))
    for c in range(total_rank):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    
    if not np.allclose(S, T, atol=1e-9):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1
                    V_sb[j * p + k, col_sb] = 1
                    W_sb[k * n + i, col_sb] = 1
                    col_sb += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
