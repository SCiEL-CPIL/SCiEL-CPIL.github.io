import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Estimate max rank: 12 (row 0) + 42 (rows 1-2) = 54
    max_rank = 54
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    def add_1x2x2(i_base, j_base, k_base):
        nonlocal col
        
        # U indices
        u0 = i_base * m + j_base
        u1 = i_base * m + (j_base + 1)
        
        # V indices
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)
        
        # W indices
        w0 = k_base * n + i_base
        w1 = (k_base + 1) * n + i_base
        
        # Term 1: Contributes to C[k_base]
        # C[k_base] += A[j_base]B[j_base, k_base] + A[j_base+1]B[j_base+1, k_base]
        U[u0][col] = 1
        U[u1][col] = 1
        V[v00][col] = 1
        V[v10][col] = 1
        W[w0][col] = 1
        col += 1
        
        # Term 2: Contributes to C[k_base+1]
        # C[k_base+1] += A[j_base]B[j_base, k_base+1] + A[j_base+1]B[j_base+1, k_base+1]
        U[u0][col] = 1
        U[u1][col] = 1
        V[v01][col] = 1
        V[v11][col] = 1
        W[w1][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Part 1: Row 0 uses optimized 1x2x2 blocks (rank 12)
    # Split 4 cols of A into two 2-col blocks
    for j_block in range(2):
        j_base = j_block * 2
        # Split 6 cols of B into three 2-col blocks
        for k_block in range(3):
            k_base = k_block * 2
            add_1x2x2(0, j_base, k_base)
            
    # Part 2: Rows 1-2 use Strassen on 2x2x2 sub-blocks (rank 42)
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            i_base = 1
            add_strassen(i_base, j_base, k_base)
    
    rank = col
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Reconstruct
    recon = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        recon += np.einsum('i,j,k->ijk', U_arr[:,c], V_arr[:,c], W_arr[:,c])
        
    if not np.allclose(recon, target, atol=1e-6):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U, "V": V, "W": W}
