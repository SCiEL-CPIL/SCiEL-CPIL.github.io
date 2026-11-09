"""Matrix multiplication 2x3x9: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time

def construct():
    n, m, p = 2, 3, 9
    rank = 50  # 32 (for 2x2x9) + 18 (for 2x1x9)
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 27 x rank
    W = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    def add_term(U_entries, V_entries, W_entries, c_idx):
        for (u_idx, u_val) in U_entries:
            U[u_idx][c_idx] += u_val
        for (v_idx, v_val) in V_entries:
            V[v_idx][c_idx] += v_val
        for (w_idx, w_val) in W_entries:
            W[w_idx][c_idx] += w_val
    
    c = 0
    
    # --- Part A: 2x2x9 (rank 32) ---
    # For k in {0..7}: 4 blocks of 2, each using Strassen (rank 7)
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # M1 = (a00+a11) * (b0k0+b0k1+b1k0+b1k1)
        add_term(
            [(0*m+0, 1), (1*m+1, 1)],
            [(0*p+k0, 1), (0*p+k1, 1), (1*p+k0, 1), (1*p+k1, 1)],
            [(k0*n+0, 1), (k0*n+1, 1), (k1*n+0, 1), (k1*n+1, 1)],
            c
        )
        c += 1
        
        # M2 = (a10+a11) * b0k0
        add_term(
            [(1*m+0, 1), (1*m+1, 1)],
            [(0*p+k0, 1)],
            [(k0*n+1, 1), (k1*n+1, -1)],
            c
        )
        c += 1
        
        # M3 = a00 * (b1k1 - b0k1)
        add_term(
            [(0*m+0, 1)],
            [(1*p+k1, 1), (0*p+k1, -1)],
            [(k1*n+0, 1), (k1*n+1, 1)],
            c
        )
        c += 1
        
        # M4 = a11 * (b0k0 - b1k0)
        add_term(
            [(1*m+1, 1)],
            [(0*p+k0, 1), (1*p+k0, -1)],
            [(k0*n+0, -1), (k0*n+1, 1)],
            c
        )
        c += 1
        
        # M5 = (a00+a01) * b1k1
        add_term(
            [(0*m+0, 1), (0*m+1, 1)],
            [(1*p+k1, 1)],
            [(k0*n+0, 1), (k1*n+0, 1)],
            c
        )
        c += 1
        
        # M6 = (a10-a00) * (b0k0+b0k1)
        add_term(
            [(1*m+0, 1), (0*m+0, -1)],
            [(0*p+k0, 1), (0*p+k1, 1)],
            [(k1*n+1, 1)],
            c
        )
        c += 1
        
        # M7 = (a01-a11) * (b1k0+b1k1)
        add_term(
            [(0*m+1, 1), (1*m+1, -1)],
            [(1*p+k0, 1), (1*p+k1, 1)],
            [(k0*n+0, -1)],
            c
        )
        c += 1
    
    # For k=8: 2x2x1 (rank 4) - schoolbook
    k8 = 8
    # C[0,8] += a[0,0]*b[0,8]
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # C[0,8] += a[0,1]*b[1,8]
    add_term([(0*m+1, 1)], [(1*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # C[1,8] += a[1,0]*b[0,8]
    add_term([(1*m+0, 1)], [(0*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1
    # C[1,8] += a[1,1]*b[1,8]
    add_term([(1*m+1, 1)], [(1*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1
    
    # --- Part B: 2x1x9 (rank 18) ---
    # C[i,k] += a[i,2] * b[2,k] for all i in {0,1}, k in {0..8}
    for i in range(2):
        for k in range(9):
            add_term(
                [(i*m+2, 1)],
                [(2*p+k, 1)],
                [(k*n+i, 1)],
                c
            )
            c += 1
    
    return {"U": U, "V": V, "W": W}
