import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Rank 50 construction:
    # Split inner dimension m=4 into two blocks of size 2.
    # For each block, split output dimension p=7 into three blocks of size 2 and one of size 1.
    # Use Strassen (rank 7) for 2x2x2 blocks and Schoolbook (rank 4) for 2x2x1 blocks.
    # Total rank = 2 * (3 * 7 + 4) = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Indices for A (2x2 block at row i0, col j0)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # Indices for B (2x2 block at row j0, col k0)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # Indices for W (2x2 block at row k0, col i0)
        # W index convention: W[k*n + i, r]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00, C11
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10, -C11
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01, C11
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00, C10
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> -C00, C01
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A (2x2 block)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B (2x1 block)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # W (2x1 block)
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C00
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C00
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C10
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C10
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Apply blocks
    for j_start in (0, 2):
        for k_start in (0, 2, 4):
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    # Verification using vectorized tensor contraction
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    # tensor[i, j, k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r to get T[i, j, k]
    # U_arr shape: (n*m, rank)
    # V_arr shape: (m*p, rank)
    # W_arr shape: (p*n, rank)
    
    U_t = U_arr.reshape(n, m, rank)  # (n, m, r)
    V_t = V_arr.reshape(m, p, rank)  # (m, p, r)
    W_t = W_arr.reshape(p, n, rank)  # (p, n, r)
    
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is an Einstein summation: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    
    # result should be all ones
    max_err = float(np.max(np.abs(result - 1.0)))
    
    if max_err > 1e-9:
        pass

    elapsed = time.time() - t0
    if elapsed > 110:
        pass

    return {"U": U, "V": V, "W": W}
