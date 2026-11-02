import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = 2, 3, 9

    # Build the rank-50 decomposition (Strassen on 2x2 blocks, (2+1) split)
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def full_decomp(Um, Vm, Wm):
        return np.einsum('ir,jr,kr->ijk', Um, Vm, Wm)

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Verify the base rank-50 is correct
    if not np.allclose(full_decomp(U_np, V_np, W_np), T, atol=1e-9):
        return {"U": U, "V": V, "W": W}

    # For rank reduction: a column r can be dropped if there exist coefficients
    # c_s for other columns s such that for all (i,j,k):
    #   U[i,r]*V[j,r]*W[k,r] = sum_s c_s * U[i,s]*V[j,s]*W[k,s]
    # This is a system of linear equations in the c_s.
    # We build the constraint matrix: for each (i,j,k) where T[i,j,k]!=0,
    # the equation is: sum_s c_s * U[i,s]*V[j,s]*W[k,s] = U[i,r]*V[j,r]*W[k,r]
    
    # Active (i,j) pairs
    active_pairs = np.argwhere(T.any(axis=2))
    I_active = active_pairs[:, 0]
    J_active = active_pairs[:, 1]
    N_active = len(I_active)
    
    # For each active (i,j), the k values that are active
    # Build constraint matrix M where M[row, s] = U[i,s]*V[j,s]*W[k,s]
    # and rhs for column r is U[i,r]*V[j,r]*W[k,r]
    
    # Number of constraints: for each active (i,j), how many k are active?
    # T[i,j,k] = 1 when j%p == i%m and k = (j%p)*n + (i//m)
    # So for each active (i,j), exactly one k is active
    # Total constraints = N_active = 54
    
    # Build the constraint matrix
    # For each active (i,j), the active k is determined
    M_constraints = np.zeros((N_active, rank))
    for idx, (i, j) in enumerate(active_pairs):
        k = (j % p) * n + (i // m)
        for s in range(rank):
            M_constraints[idx, s] = U_np[i, s] * V_np[j, s] * W_np[k, s]
    
    # Now try dropping columns one at a time
    # For column r to be droppable: the system M[:, ~r] * c = M[:, r] must be solvable
    # with c having support only on non-dropped columns
    
    best_rank = rank
    best_mask = np.ones(rank, dtype=bool)
    
    # Try single drops
    for r in range(rank):
        if time.time() > deadline - 10:
            break
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        cols = np.where(mask)[0]
        if len(cols) == 0:
            continue
        # Solve M[:, cols] * c = M[:, r]
        # This is an overdetermined system (54 equations, 49 unknowns)
        try:
            A = M_constraints[:, cols]
            b = M_constraints[:, r]
            # Check if b is in the column space of A
            # Use least squares and check residual
            c, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            res = np.linalg.norm(A @ c - b)
            if res < 1e-8:
                # Verify with actual decomposition
                U_new = U_np[:, mask]
                V_new = V_np[:, mask]
                W_new = W_np[:, mask]
                decomp = full_decomp(U_new, V_new, W_new)
                if np.allclose(decomp, T, atol=1e-9):
                    best_rank = int(mask.sum())
                    best_mask = mask.copy()
        except Exception:
            pass
    
    # If we found a single drop, try to find more
    if best_rank < rank:
        current_mask = best_mask.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for r in range(rank):
                if time.time() > deadline - 5:
                    break
                if not current_mask[r]:
                    continue
                new_mask = current_mask.copy()
                new_mask[r] = False
                cols = np.where(new_mask)[0]
                if len(cols) == 0:
                    continue
                try:
                    A = M_constraints[:, cols]
                    b = M_constraints[:, r]
                    c, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    res = np.linalg.norm(A @ c - b)
                    if res < 1e-8:
                        U_new = U_np[:, new_mask]
                        V_new = V_np[:, new_mask]
                        W_new = W_np[:, new_mask]
                        decomp = full_decomp(U_new, V_new, W_new)
                        if np.allclose(decomp, T, atol=1e-9):
                            current_mask = new_mask
                            best_rank = int(new_mask.sum())
                            best_mask = new_mask.copy()
                            improved = True
                except Exception:
                    pass
    
    if best_rank < rank:
        U_out = U_np[:, best_mask]
        V_out = V_np[:, best_mask]
        W_out = W_np[:, best_mask]
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    return {"U": U, "V": V, "W": W}
