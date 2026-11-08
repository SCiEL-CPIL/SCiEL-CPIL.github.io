import numpy as np
import time

DIMENSIONS = [2, 4, 7]
START_TIME = time.time()
DEADLINE = 145.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Calculate required rank:
    # 6 Strassen 2x2x2 blocks (j0 in {0,2}, k0 in {0,2,4}) * 7 = 42
    # 2 schoolbook 2x2x1 blocks (j0 in {0,2}, k0=6) * 4 = 8
    # Total = 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
    
    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
    
    # Process all blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
        add_schoolbook_221_block(0, j0, 6)
    
    # Trim to actual rank
    actual_rank = col
    U_out = [row[:actual_rank] for row in U]
    V_out = [row[:actual_rank] for row in V]
    W_out = [row[:actual_rank] for row in W]
    
    # Quick verification
    U_np = np.array(U_out, dtype=float)
    V_np = np.array(V_out, dtype=float)
    W_np = np.array(W_out, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(actual_rank):
        T += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1
    
    if not np.allclose(T, T_expected, atol=1e-10):
        # Fallback: use schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_sb[i * m + j][r] = 1
                    V_sb[j * p + k][r] = 1
                    W_sb[k * n + i][r] = 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_out, "V": V_out, "W": W_out}
