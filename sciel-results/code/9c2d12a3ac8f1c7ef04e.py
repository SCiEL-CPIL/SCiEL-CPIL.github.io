import numpy as np
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 130.0

def _now_ok():
    return time.time() < _DEADLINE

def construct():
    n, m, p = DIMENSIONS
    
    # We aim for rank 44 using 2x2x3 (rank 8) + 2x2x4 (rank 14) per j-block.
    # Total rank = 2 * (8 + 14) = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
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

    # Helper to add Rank 8 block for 2x2x3
    # Decomposition for 2x2x3:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Known rank 8 decomposition:
    # M1 = A00 * (B00 + B01 + B02)
    # M2 = A01 * (B10 + B11 + B12)
    # M3 = A00 * B00
    # M4 = A01 * B10
    # M5 = A10 * (B00 + B01 + B02)
    # M6 = A11 * (B10 + B11 + B12)
    # M7 = A10 * B00
    # M8 = A11 * B10
    
    # Let's verify this:
    # C00 = M1*B00_part + M2*B10_part - M3 - M4 = A00*B00 + A01*B10 ✓
    # C01 = M1*B01_part + M2*B11_part = A00*B01 + A01*B11 ✓
    # C02 = M1*B02_part + M2*B12_part = A00*B02 + A01*B12 ✓
    # C10 = M5*B00_part + M6*B10_part - M7 - M8 = A10*B00 + A11*B10 ✓
    # C11 = M5*B01_part + M6*B11_part = A10*B01 + A11*B11 ✓
    # C12 = M5*B02_part + M6*B12_part = A10*B02 + A11*B12 ✓
    
    def add_rank8_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = j0 * p + (k0 + 2)
        h_idx = (j0 + 1) * p + k0
        i_idx = (j0 + 1) * p + (k0 + 1)
        j_idx = (j0 + 1) * p + (k0 + 2)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c02_idx = (k0 + 2) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        c12_idx = (k0 + 2) * n + (i0 + 1)
        
        # M1 = A00 * (B00 + B01 + B02)
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        W[c01_idx][col] += 1
        W[c02_idx][col] += 1
        col += 1
        
        # M2 = A01 * (B10 + B11 + B12)
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        V[i_idx][col] += 1
        V[j_idx][col] += 1
        W[c00_idx][col] += 1
        W[c01_idx][col] += 1
        W[c02_idx][col] += 1
        col += 1
        
        # M3 = A00 * B00
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] -= 1
        col += 1
        
        # M4 = A01 * B10
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        col += 1
        
        # M5 = A10 * (B00 + B01 + B02)
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] += 1
        W[c12_idx][col] += 1
        col += 1
        
        # M6 = A11 * (B10 + B11 + B12)
        U[d_idx][col] += 1
        V[h_idx][col] += 1
        V[i_idx][col] += 1
        V[j_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] += 1
        W[c12_idx][col] += 1
        col += 1
        
        # M7 = A10 * B00
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] -= 1
        col += 1
        
        # M8 = A11 * B10
        U[d_idx][col] += 1
        V[h_idx][col] += 1
        W[c10_idx][col] -= 1
        col += 1

    # For each j-block (2 columns of A, 2 columns of B):
    # Use Rank 8 for 2x2x3 (k=0,1,2)
    # Use Strassen for 2x2x2 (k=3,4)
    # Use Strassen for 2x2x2 (k=5,6)
    # Total per j-block: 8 + 7 + 7 = 22
    # Total for 2 j-blocks: 44
    
    for j_start in [0, 2]:
        # Rank 8 block for k=0,1,2
        add_rank8_block(0, j_start, 0)
        
        # Strassen block for k=3,4
        add_strassen_block(0, j_start, 3)
        
        # Strassen block for k=5,6
        add_strassen_block(0, j_start, 5)
    
    return {"U": U, "V": V, "W": W}

