"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_strassen_schoolbook(n, m, p):
    """
    Build the known rank-50 decomposition: 
    6 Strassen (2x2x2) blocks + 2 schoolbook (2x2x1) blocks.
    
    Structure:
    A (2x4) is split into A_left (cols 0,1) and A_right (cols 2,3).
    B (4x7) is split into B_top (rows 0,1) and B_bot (rows 2,3).
    
    C = A_left @ B_top + A_right @ B_bot
    
    For A_left @ B_top (2x2 @ 2x7):
    - Split B_top cols into (0,1), (2,3), (4,5), (6).
    - First three are 2x2x2 Strassen blocks.
    - Last is 2x2x1 Schoolbook block.
    Total: 3 * 7 + 4 = 25.
    
    For A_right @ B_bot (2x2 @ 2x7):
    - Same structure.
    Total: 25.
    
    Total rank: 50.
    """
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Indices for A (2x2 block)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # Indices for B (2x2 block)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # Indices for C (2x2 block)
        # Convention: W[k*n + i, r]
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> C11 += 1
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> C00 += 1
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
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
        
        # C (2x1 block)
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

        # P2 = b*g
        U[b_idx, col] += 1
        V[g_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

        # P3 = c*e
        U[c_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

        # P4 = d*g
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

    # Apply blocks for A_left (j0=0) @ B_top (j0=0)
    # k0 in {0, 2, 4} for Strassen, k0=6 for Schoolbook
    for k0 in [0, 2, 4]:
        add_strassen_block(0, 0, k0)
    add_schoolbook_221_block(0, 0, 6)

    # Apply blocks for A_right (j0=2) @ B_bot (j0=2)
    # k0 in {0, 2, 4} for Strassen, k0=6 for Schoolbook
    for k0 in [0, 2, 4]:
        add_strassen_block(0, 2, k0)
    add_schoolbook_221_block(0, 2, 6)

    return U, V, W, col

def _verify(U, V, W, n, m, p):
    """Verify the decomposition reproduces the all-ones tensor."""
    U_t = U.reshape(n, m, -1)
    V_t = V.reshape(m, p, -1)
    W_t = W.reshape(p, n, -1)
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    T_check = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    expected = np.ones((n, m, p), dtype=np.int64)
    return np.array_equal(T_check, expected)

def _attempt_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by removing redundant terms.
    We iterate through columns and try to remove them if the remaining
    set still forms a valid decomposition.
    """
    current_rank = U.shape[1]
    U_work, V_work, W_work = U.copy(), V.copy(), W.copy()
    rank_work = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Pre-compute expected tensor for faster checks if needed, 
    # but einsum is fast enough for rank ~50.
    
    # Greedy removal
    # We repeat passes until no more removals are possible or time is up.
    # In each pass, we try to remove one column.
    
    while time.time() < deadline - 5:
        removed_any = False
        for r in range(rank_work):
            if time.time() > deadline - 5:
                break
            
            # Create mask to exclude column r
            mask = np.ones(rank_work, dtype=bool)
            mask[r] = False
            
            U_new = U_work[:, mask]
            V_new = V_work[:, mask]
            W_new = W_work[:, mask]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                # Removal successful
                U_work, V_work, W_work = U_new, V_new, W_new
                rank_work = U_work.shape[1]
                best_U, best_V, best_W = U_work.copy(), V_work.copy(), W_work.copy()
                best_rank = rank_work
                removed_any = True
                break # Restart loop with new rank
        
        if not removed_any:
            break
            
    if best_rank < current_rank:
        return (best_U, best_V, best_W)
    return None

def construct():
    start_time = time.time()
    # 120s limit, keep 10s buffer for safety
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # 1. Build initial rank-50 decomposition
    U, V, W, col = _build_strassen_schoolbook(n, m, p)
    
    # Verify initial construction
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook (rank 56) if construction failed (should not happen)
        return _schoolbook(n, m, p)

    # 2. Attempt rank reduction
    if time.time() < deadline - 20:
        try:
            reduced = _attempt_rank_reduction(U, V, W, n, m, p, deadline)
            if reduced is not None:
                r_red = reduced[0].shape[1]
                if r_red < col:
                    return {"U": reduced[0].tolist(), "V": reduced[1].tolist(), "W": reduced[2].tolist()}
        except Exception:
            pass

    # 3. Return the best found (likely rank 50)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
