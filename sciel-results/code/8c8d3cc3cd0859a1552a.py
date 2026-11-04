"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _build_optimized(n, m, p):
    """
    Build a rank-36 decomposition for 2x3 * 3x10.
    - 5 Strassen blocks for the 2x2 * 2x10 part (rank 35).
    - 1 Outer product term for the 2x1 * 1x10 part (rank 1).
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # --- Part 1: A_left (2x2) @ B_top (2x10) via Strassen ---
    # Strassen coefficients for 2x2x2
    # U (A) indices: 0:a00, 1:a01, 2:a10, 3:a11 -> Global: 0, 1, 3, 4
    # V (B) indices: 0:b00, 1:b01, 2:b10, 3:b11 -> Global: 2b, 2b+1, 10+2b, 10+2b+1
    # W (C) indices: 0:c00, 1:c01, 2:c10, 3:c11 -> Global: 2b, 2b+1, 2b+2, 2b+3 (k*n+i)
    
    # Local indices for Strassen steps
    # L[s]: coeffs for a_ij
    # R[s]: coeffs for b_jk
    # Wc[s]: coeffs for c_ik
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    L = [
        {0: 1, 3: 1},       # M1: a00, a11
        {2: 1, 3: 1},       # M2: a10, a11
        {0: 1},             # M3: a00
        {3: 1},             # M4: a11
        {0: 1, 1: 1},       # M5: a00, a01
        {2: 1, 0: -1},      # M6: a10, -a00
        {1: 1, 3: -1}       # M7: a01, -a11
    ]
    
    R = [
        {0: 1, 3: 1},       # M1: b00, b11
        {0: 1},             # M2: b00
        {1: 1, 3: -1},      # M3: b01, -b11
        {2: 1, 0: -1},      # M4: b10, -b00
        {3: 1},             # M5: b11
        {0: 1, 1: 1},       # M6: b00, b01
        {2: 1, 3: 1}        # M7: b10, b11
    ]
    
    Wc = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            # Fill U (A indices)
            # Local idx 0->a00(i0,j0), 1->a01(i0,j1), 2->a10(i1,j0), 3->a11(i1,j1)
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j][r] += val
            
            # Fill V (B indices)
            # Local idx 0->b00(j0,k0), 1->b01(j0,k1), 2->b10(j1,k0), 3->b11(j1,k1)
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k_local = local_idx % 2
                V[j * p + k_offset + k_local][r] += val
            
            # Fill W (C indices)
            # Local idx 0->c00(i0,k0), 1->c01(i0,k1), 2->c10(i1,k0), 3->c11(i1,k1)
            for local_idx, val in Wc[s].items():
                i = local_idx // 2
                k_local = local_idx % 2
                W[(k_offset + k_local) * n + i][r] += val
            
            r += 1
    
    # --- Part 2: A_right (2x1) @ B_bottom (1x10) via Outer Product ---
    # This part computes A_i2 * B_2k
    # We can do this with a single rank term.
    # U[i, 2, r] = 1 for i=0,1
    # V[2, k, r] = 1 for k=0..9
    # W[k, i, r] = 1 for k=0..9, i=0,1
    
    # U indices: i*m + j. j=2. i=0 -> 2, i=1 -> 5.
    U[0 * m + 2][r] = 1
    U[1 * m + 2][r] = 1
    
    # V indices: j*p + k. j=2. k=0..9 -> 20..29.
    for k in range(p):
        V[2 * p + k][r] = 1
        
    # W indices: k*n + i. k=0..9, i=0,1.
    for k in range(p):
        W[k * n + 0][r] = 1
        W[k * n + 1][r] = 1
        
    r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_optimized(n, m, p)

    if not _validate(n, m, p, U, V, W):
        U, V, W, rank = _build_trivial(n, m, p)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
