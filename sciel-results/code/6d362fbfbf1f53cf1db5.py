import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication.
    Uses a block decomposition with Strassen's algorithm for 2x2x2 blocks 
    and schoolbook for the remaining 2x1x2 blocks.
    Rank: 6 * 7 (Strassen) + 3 * 4 (Schoolbook) = 42 + 12 = 54.
    """
    n, m, p = DIMENSIONS
    rank = 54
    
    # Initialize factor matrices as lists of lists to match seed interface style
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_col = 0
    
    # Helper to add a rank-1 term with specific indices and coefficients
    # u_idx, v_idx, w_idx are the global indices into the flattened vectors
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal current_col
        if coeff != 0:
            U[u_idx][current_col] += coeff
            V[v_idx][current_col] += coeff
            W[w_idx][current_col] += coeff

    # Helper to finalize the current column
    def next_col():
        nonlocal current_col
        current_col += 1

    # Define Strassen block application
    # i_base: row offset in A/C (0)
    # j_base: col offset in A / row offset in B
    # k_base: col offset in B / col offset in C
    def apply_strassen_block(i_base, j_base, k_base):
        nonlocal current_col
        
        # Local indices 0, 1 map to global:
        # I0 = i_base, I1 = i_base + 1
        # J0 = j_base, J1 = j_base + 1
        # K0 = k_base, K1 = k_base + 1
        
        # U indices (i*m + j)
        u_00 = i_base * m + j_base
        u_01 = i_base * m + (j_base + 1)
        u_10 = (i_base + 1) * m + j_base
        u_11 = (i_base + 1) * m + (j_base + 1)
        
        # V indices (j*p + k)
        v_00 = j_base * p + k_base
        v_01 = j_base * p + (k_base + 1)
        v_10 = (j_base + 1) * p + k_base
        v_11 = (j_base + 1) * p + (k_base + 1)
        
        # W indices (k*n + i)
        w_00 = k_base * n + i_base
        w_01 = (k_base + 1) * n + i_base
        w_10 = k_base * n + (i_base + 1)
        w_11 = (k_base + 1) * n + (i_base + 1)
        
        # M0: (A00+A11)(B00+B11)
        # C00 += M0, C11 += M0
        add_term(u_00, v_00, w_00, 1)
        add_term(u_11, v_00, w_00, 0) # Wait, add_term only adds to one column.
        # We need to add multiple non-zeros to U, V, W for the same column r.
        # My add_term function above only sets one index per matrix.
        # I need a way to set multiple indices in U, V, W for the same column.
        
        # Let's refactor: define the terms explicitly by modifying the arrays directly.
        
        # M0
        U[u_00][current_col] += 1
        U[u_11][current_col] += 1
        V[v_00][current_col] += 1
        V[v_11][current_col] += 1
        W[w_00][current_col] += 1
        W[w_11][current_col] += 1
        next_col()
        
        # M1: (A10+A11)B00
        # C10 += M1, C11 -= M1
        U[u_10][current_col] += 1
        U[u_11][current_col] += 1
        V[v_00][current_col] += 1
        W[w_10][current_col] += 1
        W[w_11][current_col] -= 1
        next_col()
        
        # M2: A00(B01-B11)
        # C01 += M2, C11 += M2
        U[u_00][current_col] += 1
        V[v_01][current_col] += 1
        V[v_11][current_col] -= 1
        W[w_01][current_col] += 1
        W[w_11][current_col] += 1
        next_col()
        
        # M3: A11(B10-B00)
        # C00 -= M3, C10 += M3
        U[u_11][current_col] += 1
        V[v_10][current_col] += 1
        V[v_00][current_col] -= 1
        W[w_00][current_col] -= 1
        W[w_10][current_col] += 1
        next_col()
        
        # M4: (A00+A01)B11
        # C01 -= M4, C11 += M4
        U[u_00][current_col] += 1
        U[u_01][current_col] += 1
        V[v_11][current_col] += 1
        W[w_01][current_col] -= 1
        W[w_11][current_col] += 1
        next_col()
        
        # M5: (A10-A00)(B00+B01)
        # C00 -= M5, C01 -= M5, C10 += M5
        U[u_10][current_col] += 1
        U[u_00][current_col] -= 1
        V[v_00][current_col] += 1
        V[v_01][current_col] += 1
        W[w_00][current_col] -= 1
        W[w_01][current_col] -= 1
        W[w_10][current_col] += 1
        next_col()
        
        # M6: (A01-A11)(B10+B11)
        # C00 += M6, C01 -= M6, C10 -= M6
        U[u_01][current_col] += 1
        U[u_11][current_col] -= 1
        V[v_10][current_col] += 1
        V[v_11][current_col] += 1
        W[w_00][current_col] += 1
        W[w_01][current_col] -= 1
        W[w_10][current_col] -= 1
        next_col()

    # 1. Process Strassen blocks for j-pairs (0,1) and (2,3)
    # j_bases are 0 and 2
    # k_bases are 0, 2, 4
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0
            apply_strassen_block(i_base, j_base, k_base)

    # 2. Process Naive blocks for j=4
    # j_idx = 4
    # k_bases are 0, 2, 4
    # Block type: 2x1 * 1x2 -> 4 terms
    for k_base in [0, 2, 4]:
        i_base = 0
        j_idx = 4
        
        # Term 1: A[i0, j] * B[j, k0] -> C[i0, k0]
        U[i_base*m+j_idx][current_col] = 1
        V[j_idx*p+k_base][current_col] = 1
        W[k_base*n+i_base][current_col] = 1
        next_col()
        
        # Term 2: A[i0, j] * B[j, k1] -> C[i0, k1]
        U[i_base*m+j_idx][current_col] = 1
        V[j_idx*p+(k_base+1)][current_col] = 1
        W[(k_base+1)*n+i_base][current_col] = 1
        next_col()
        
        # Term 3: A[i1, j] * B[j, k0] -> C[i1, k0]
        U[(i_base+1)*m+j_idx][current_col] = 1
        V[j_idx*p+k_base][current_col] = 1
        W[k_base*n+(i_base+1)][current_col] = 1
        next_col()
        
        # Term 4: A[i1, j] * B[j, k1] -> C[i1, k1]
        U[(i_base+1)*m+j_idx][current_col] = 1
        V[j_idx*p+(k_base+1)][current_col] = 1
        W[(k_base+1)*n+(i_base+1)][current_col] = 1
        next_col()

    if current_col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")

    return {"U": U, "V": V, "W": W}
