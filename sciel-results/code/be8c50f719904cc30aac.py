import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation:
    # 2 blocks of A (size 2x2)
    # For each A block, B has 7 columns.
    # Split B columns into 3 blocks of 2 (Strassen) and 1 block of 1 (Schoolbook).
    # Strassen 2x2x2 rank is 7.
    # Schoolbook 2x2x1 rank is 4.
    # Rank per A block = 3 * 7 + 1 * 4 = 25.
    # Total rank = 2 * 25 = 50.
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        # A indices
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # C (W) indices: W[k*n + i, r]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
        U[a_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += M2, C11 -= M2
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += M3, C11 += M3
        U[a_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] -= 1
        W[c01_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += M4, C10 += M4
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        V[e_idx, col] -= 1
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= M5, C01 += M5
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1
        W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += M6
        U[c_idx, col] += 1
        U[a_idx, col] -= 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += M7
        U[b_idx, col] += 1
        U[d_idx, col] -= 1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    # Helper to add Schoolbook 2x2x1 block
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
        
        # P1 = a*e -> C00 += P1
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        # P2 = b*g -> C00 += P2
        U[b_idx, col] += 1
        V[g_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        # P3 = c*e -> C10 += P3
        U[c_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        # P4 = d*g -> C10 += P4
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

    # Process blocks
    # A is 2x4. Split into two 2x2 blocks along columns: j=0,1 and j=2,3.
    # B is 4x7. For each A block, we multiply by the corresponding B block (4x7).
    # We split the 7 columns of B into 2,2,2,1.
    
    for j_start in [0, 2]:
        # Strassen blocks for k=0,1; k=2,3; k=4,5
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        # Schoolbook block for k=6
        add_schoolbook_221_block(0, j_start, 6)
        
    # Verify col is 50
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
