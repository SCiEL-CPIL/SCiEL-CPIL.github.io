import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 2, 3, 9
    n_rows, m_rows, p_rows = n * m, m * p, p * n  # 6, 27, 18

    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n_rows)]
    V = [[0] * rank for _ in range(m_rows)]
    W = [[0] * rank for _ in range(p_rows)]

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

    # Identify active (i,j) pairs
    active_pairs = np.argwhere(T.any(axis=2))
    I_active = active_pairs[:, 0]
    J_active = active_pairs[:, 1]
    N = len(I_active)

    # Build compressed matrix for null space detection
    comp = np.zeros((N * (p*n), rank))
    for r in range(rank):
        uv = U_np[I_active, r] * V_np[J_active, r]
        flat = np.zeros(N*(p*n))
        for k in range(p*n):
            flat[k::(p*n)] = uv * W_np[k, r]
        comp[:, r] = flat

    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = rank

    try:
        # SVD to find null space
        _, s, Vt = np.linalg.svd(comp, full_matrices=False)
        tol = s[0] * rank * 1e-8
        num_rank = int(np.sum(s > tol))
        null_dim = rank - num_rank

        if null_dim <= 0:
            # No null space found, try direct column dropping
            mask = np.ones(rank, dtype=bool)
            for c in range(rank):
                if time.time() > deadline - 10:
                    break
                mask[c] = False
                U_new = U_np[:, mask]
                V_new = V_np[:, mask]
                W_new = W_np[:, mask]
                res = np.sum((full_decomp(U_new, V_new, W_new) - T)**2)
                if res < 1e-8:
                    mask[c] = True
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = int(mask.sum())
                    mask[c] = False
                else:
                    mask[c] = True
        else:
            # Get null space vectors
            null_vecs = [Vt[rank - d - 1] for d in range(null_dim)]
            
            # Greedy removal: remove columns with highest null-space weight
            mask = np.ones(rank, dtype=bool)
            
            # Sort columns by their max null-space weight
            null_weights = np.zeros(rank)
            for nv in null_vecs:
                null_weights = np.maximum(null_weights, np.abs(nv))
            
            # Try removing columns in order of decreasing null weight
            order = np.argsort(-null_weights)
            
            for c in order:
                if time.time() > deadline - 10:
                    break
                if not mask[c]:
                    continue
                mask[c] = False
                U_new = U_np[:, mask]
                V_new = V_np[:, mask]
                W_new = W_np[:, mask]
                res = np.sum((full_decomp(U_new, V_new, W_new) - T)**2)
                if res < 1e-8:
                    mask[c] = True
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = int(mask.sum())
                    mask[c] = False
                else:
                    mask[c] = True

            # Second pass: try removing 2 columns at a time from remaining
            if best_rank < rank:
                remaining = [c for c in range(rank) if mask[c]]
                for c1, c2 in itertools.combinations(remaining, 2):
                    if time.time() > deadline - 5:
                        break
                    m2 = mask.copy()
                    m2[c1] = False
                    m2[c2] = False
                    U_new = U_np[:, m2]
                    V_new = V_np[:, m2]
                    W_new = W_np[:, m2]
                    res = np.sum((full_decomp(U_new, V_new, W_new) - T)**2)
                    if res < 1e-8:
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = int(m2.sum())
                        break
    except Exception:
        pass

    # Final verification with tighter tolerance
    if best_rank < rank:
        final_res = np.sum((full_decomp(best_U, best_V, best_W) - T)**2)
        if final_res < 1e-8:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U, "V": V, "W": W}
