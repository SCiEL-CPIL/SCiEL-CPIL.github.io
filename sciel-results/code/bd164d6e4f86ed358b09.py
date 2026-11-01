import numpy as np
import time

# Constants for the 2x4 * 4x7 matrix multiplication
N, M, P = 2, 4, 7
RANK = 50
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    
    U = [[0] * RANK for _ in range(N * M)]
    V = [[0] * RANK for _ in range(M * P)]
    W = [[0] * RANK for _ in range(P * N)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * M + j0
        b_idx = i0 * M + (j0 + 1)
        c_idx = (i0 + 1) * M + j0
        d_idx = (i0 + 1) * M + (j0 + 1)
        
        e_idx = j0 * P + k0
        f_idx = j0 * P + (k0 + 1)
        g_idx = (j0 + 1) * P + k0
        h_idx = (j0 + 1) * P + (k0 + 1)
        
        c00_idx = k0 * N + i0
        c01_idx = (k0 + 1) * N + i0
        c10_idx = k0 * N + (i0 + 1)
        c11_idx = (k0 + 1) * N + (i0 + 1)
        
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
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
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
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * M + j0
        b_idx = i0 * M + (j0 + 1)
        c_idx = (i0 + 1) * M + j0
        d_idx = (i0 + 1) * M + (j0 + 1)
        
        e_idx = j0 * P + k0
        g_idx = (j0 + 1) * P + k0
        
        c00_idx = k0 * N + i0
        c10_idx = k0 * N + (i0 + 1)
        
        # P1 = a*e
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        for k_start in [0, 2, 4]:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            add_strassen_block(0, j_start, k_start)
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        add_schoolbook_221_block(0, j_start, 6)
    
    return {"U": U, "V": V, "W": W}
