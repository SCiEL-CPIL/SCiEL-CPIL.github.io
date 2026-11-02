import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Strategy: Build a decomposition and then try to reduce rank
    # using null space analysis and iterative column removal

    # Start with a naive decomposition (rank = 54)
    # Each term corresponds to one (am, bm) pair where am==bj
    rank_naive = m * p  # 27 terms... wait, let me reconsider

    # Actually, the naive decomposition: for each middle index am (0..2),
    # and for each column bm (0..8), we have one term
    # That gives 3*9 = 27 terms, but each term contributes to 2 outputs
    # Let me think more carefully.

    # The tensor T[i,j,k] = 1 iff i = am*m + ai, j = bm*p + am, k = bm*n + ai
    # for some am, ai, bm.
    # 
    # A rank-1 term is U[i]*V[j]*W[k]
    # For a fixed (am, bm, ai), the term is:
    #   U[am*m+ai] = 1, V[bm*p+am] = 1, W[bm*n+ai] = 1
    # This gives 3*9*2 = 54 terms for the naive decomposition.
    #
    # But we can do better by grouping.

    # Let me try the Strassen-based approach from the current program
    # and then attempt rank reduction.

    # Build rank-50 decomposition (same as current)
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1

        U[u_c, col+1] += 1
        U[u_d, col+1] += 1
        V[v_e, col+1] += 1

        U[u_a, col+2] += 1
        V[v_f, col+2] += 1
        V[v_h, col+2] -= 1

        U[u_d, col+3] += 1
        V[v_g, col+3] += 1
        V[v_e, col+3] -= 1

        U[u_a, col+4] += 1
        U[u_b, col+4] += 1
        V[v_h, col+4] += 1

        U[u_c, col+5] += 1
        U[u_a, col+5] -= 1
        V[v_e, col+5] += 1
        V[v_f, col+5] += 1

        U[u_b, col+6] += 1
        U[u_d, col+6] -= 1
        V[v_g, col+6] += 1
        V[v_h, col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00, col] += 1
        W[w_00, col+3] += 1
        W[w_00, col+4] -= 1
        W[w_00, col+6] += 1

        W[w_01, col+2] += 1
        W[w_01, col+4] += 1

        W[w_10, col+1] += 1
        W[w_10, col+3] += 1

        W[w_11, col] += 1
        W[w_11, col+1] -= 1
        W[w_11, col+2] += 1
        W[w_11, col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a, col] += 1
    V[v_e, col] += 1
    U[u_b, col+1] += 1
    V[v_g, col+1] += 1
    U[u_c, col+2] += 1
    V[v_e, col+2] += 1
    U[u_d, col+3] += 1
    V[v_g, col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08, col] += 1
    W[w_08, col+1] += 1
    W[w_18, col+2] += 1
    W[w_18, col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0, col] += 1
        V[v_b, col] += 1
        W[k * n + 0, col] += 1
        U[u_ar1, col+1] += 1
        V[v_b, col+1] += 1
        W[k * n + 1, col+1] += 1
        col += 2

    # Verify the decomposition
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        # Fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank using greedy column removal
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank

    while True:
        if time.time() > deadline - 20:
            break
            
        improved = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 20:
                break
                
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            
            # Check if decomposition still holds
            decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            residual = np.sum((decomp_new - T)**2)
            
            if residual < 1e-8:
                # Accept removal
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break # Restart loop with new rank
                
        if not improved:
            break
            
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
