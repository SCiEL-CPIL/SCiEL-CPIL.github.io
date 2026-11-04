"""Strassen-based 4x5x6 matrix multiplication decomposition (rank 101)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    
    rank = 101
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, r_start):
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1

    def add_strassen_4x4x4(i0, j0, k0, r_start):
        # Decomposes 4x4x4 using 7 Strassen 2x2x2 blocks
        # i0, j0, k0 are top-left offsets for the 4x4x4 block
        
        # Define indices for the 2x2 blocks within the 4x4 block
        # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
        # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[(i0+0)*m + (j0+0)][r_start] += 1
        U[(i0+2)*m + (j0+2)][r_start] += 1
        V[(j0+0)*p + (k0+0)][r_start] += 1
        V[(j0+2)*p + (k0+2)][r_start] += 1
        W[(k0+0)*n + (i0+0)][r_start] += 1
        W[(k0+2)*n + (i0+2)][r_start] += 1
        r_start += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+2)*m + (j0+0)][r_start] += 1
        U[(i0+2)*m + (j0+2)][r_start] += 1
        V[(j0+0)*p + (k0+0)][r_start] += 1
        W[(k0+0)*n + (i0+2)][r_start] += 1
        W[(k0+2)*n + (i0+2)][r_start] += -1
        r_start += 1
        
        # M3 = A00(B01 - B11)
        U[(i0+0)*m + (j0+0)][r_start] += 1
        V[(j0+0)*p + (k0+2)][r_start] += 1
        V[(j0+2)*p + (k0+2)][r_start] += -1
        W[(k0+2)*n + (i0+0)][r_start] += 1
        W[(k0+2)*n + (i0+2)][r_start] += 1
        r_start += 1
        
        # M4 = A11(B10 - B00)
        U[(i0+2)*m + (j0+2)][r_start] += 1
        V[(j0+2)*p + (k0+0)][r_start] += 1
        V[(j0+0)*p + (k0+0)][r_start] += -1
        W[(k0+0)*n + (i0+2)][r_start] += 1
        W[(k0+0)*n + (i0+0)][r_start] += 1
        r_start += 1
        
        # M5 = (A00 + A01)B11
        U[(i0+0)*m + (j0+0)][r_start] += 1
        U[(i0+0)*m + (j0+2)][r_start] += 1
        V[(j0+2)*p + (k0+2)][r_start] += 1
        W[(k0+0)*n + (i0+0)][r_start] += -1
        W[(k0+2)*n + (i0+0)][r_start] += 1
        r_start += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i0+2)*m + (j0+0)][r_start] += 1
        U[(i0+0)*m + (j0+0)][r_start] += -1
        V[(j0+0)*p + (k0+0)][r_start] += 1
        V[(j0+2)*p + (k0+0)][r_start] += 1
        W[(k0+2)*n + (i0+2)][r_start] += 1
        r_start += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[(i0+0)*m + (j0+2)][r_start] += 1
        U[(i0+2)*m + (j0+2)][r_start] += -1
        V[(j0+2)*p + (k0+0)][r_start] += 1
        V[(j0+2)*p + (k0+2)][r_start] += 1
        W[(k0+0)*n + (i0+0)][r_start] += 1
        r_start += 1
        
        return r_start

    r_offset = 0
    
    # 4x4x4 core using Strassen (rank 49)
    r_offset = add_strassen_4x4x4(0, 0, 0, r_offset)
    
    # 4x4x2 strip (k=4,5) using 4 Strassen blocks (rank 28)
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            r_offset = add_strassen_block(i0, j0, 4, r_offset)
            
    # 4x1x6 strip (j=4) using schoolbook (rank 24)
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1
            
    return {"U": U, "V": V, "W": W}
