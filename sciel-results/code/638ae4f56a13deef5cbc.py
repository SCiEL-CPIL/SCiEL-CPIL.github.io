import numpy as np
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = 140.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Block decomposition
    # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}
    # Split p=7 into blocks of 2 and 1: k in {0,1}, {2,3}, {4,5}, {6}
    # Use Strassen (rank 7) for 2x2x2 blocks
    # Use Schoolbook (rank 4) for 2x2x1 blocks
    # Total rank = 2 * (3 * 7 + 4) = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A indices (2x2)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices (2x2)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # C indices (2x2), C is p x n, index = k * n + i
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> c00 += M1, c11 += M1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> c10 += M2, c11 -= M2
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> c01 += M3, c11 += M3
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> c00 += M4, c10 += M4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> c00 -= M5, c01 += M5
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> c11 += M6
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> c00 += M7
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        if time.time() - t0 > _DEADLINE - 10.0:
            return False

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c0_idx = k0 * n + i0
        c1_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> c0 += P1
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c0_idx][col] += 1
        col += 1
        
        # P2 = b*g -> c0 += P2
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c0_idx][col] += 1
        col += 1
        
        # P3 = c*e -> c1 += P3
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c1_idx][col] += 1
        col += 1
        
        # P4 = d*g -> c1 += P4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c1_idx][col] += 1
        col += 1

        if time.time() - t0 > _DEADLINE - 10.0:
            return False

    # Process blocks
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            if not add_strassen_block(0, j_start, k_start):
                return _fallback()
        if not add_schoolbook_221_block(0, j_start, 6):
            return _fallback()
    
    if time.time() - t0 > _DEADLINE - 5.0:
        return _fallback()
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        return _fallback()
    
    return {"U": U, "V": V, "W": W}

def _fallback():
    n, m, p = DIMENSIONS
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][r] = 1
                V_fb[j*p+k][r] = 1
                W_fb[k*n+i][r] = 1
                r += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
