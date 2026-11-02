import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

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

    # Verify the base rank-50 is correct (cheap)
    if not np.allclose(full_decomp(U_np, V_np, W_np), T, atol=1e-9):
        return {"U": U, "V": V, "W": W}

    # Identify which (i,j) pairs are "active" (have T[i,j,:]!=0)
    active_pairs = np.argwhere(T.any(axis=2))  # shape (N,2)
    I_active = active_pairs[:, 0]
    J_active = active_pairs[:, 1]

    # Build the compressed linear-dependency matrix:
    # For each rank column r, the "signature" restricted to active (i,j) is
    #   s_r = (U[i,r]*V[j,r])_{(i,j) in active}  (length N)
    # The full tensor column is sum over k of s_r[i,j]*W[k,r] * e_{i,j,k}.
    # Since W has distinct patterns, we capture dependencies by stacking
    # s_r * W[k,r] for the active (i,j) and each k. To keep it small we use
    # the per-(i,j) signature times the W vector.
    N = len(I_active)
    # compressed[r, idx] where idx = ((i,j index)* (p*n) + k)
    # But that's N*(p*n) rows = up to 54*18=972 rows x 50 -> fine
    comp = np.zeros((N * (p*n), rank))
    for r in range(rank):
        uv = U_np[I_active, r] * V_np[J_active, r]  # length N
        # for each k, value at (i,j,k) = uv[ij] * W[k,r]
        comp[r*0 + np.arange(N*(p*n))] = 0  # placeholder
        flat = np.zeros(N*(p*n))
        for k in range(p*n):
            flat[k::(p*n)] = uv * W_np[k, r]
        comp[:, r] = flat

    # SVD of compressed matrix to find null space
    best_U = U_np
    best_V = V_np
    best_W = W_np
    best_rank = rank
    best_residual = 0.0

    try:
        _, s, Vt = np.linalg.svd(comp, full_matrices=False)
        tol = s[0] * rank * 1e-9
        num_rank = int(np.sum(s > tol))
        null_dim = rank - num_rank
        if null_dim <= 0:
            null_dim = 1
            null_vecs = [Vt[-1]]
        else:
            null_vecs = [Vt[-(d+1)] for d in range(null_dim)]

        # Greedily drop columns that lie in the null space.
        # A column r can be dropped if its removal is "covered" by a null-space
        # combination; equivalently, we try dropping the column with the largest
        # weight in some null vector and verify the residual.
        mask = np.ones(rank, dtype=bool)
        current_U, current_V, current_W = U_np.copy(), V_np.copy(), W_np.copy()

        # Verify base residual is ~0
        base_res = np.sum((full_decomp(current_U, current_V, current_W) - T)**2)
        best_residual = base_res

        # Try single-column drops guided by null space
        drop_candidates = []
        for nv in null_vecs:
            abs_nv = np.abs(nv)
            thr = np.max(abs_nv) * 0.05
            for c in np.where(abs_nv > thr)[0]:
                drop_candidates.append(c)
        # Also try all columns as a fallback (cheap enough: 50 verifications)
        drop_candidates = list(set(drop_candidates)) + list(range(rank))
        drop_candidates = list(dict.fromkeys(drop_candidates))

        for c in drop_candidates:
            if time.time() > deadline - 8:
                break
            if not mask[c]:
                continue
            mask[c] = False
            U_new = U_np[:, mask]
            V_new = V_np[:, mask]
            W_new = W_np[:, mask]
            res = np.sum((full_decomp(U_new, V_new, W_new) - T)**2)
            if res < 1e-8:
                mask[c] = True  # restore for further search
                current_U, current_V, current_W = U_new, V_new, W_new
                mask[c] = False
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = int(mask.sum())
                best_residual = res
                # continue to try dropping more
            else:
                mask[c] = True  # restore

        # Try dropping 2 columns at a time (from the null-space guided set)
        if best_rank < rank:
            guided = [c for c in drop_candidates if abs(max(null_vecs, key=lambda v: abs(v[c]))[c]) > 0.05*np.max(null_vecs[0])]
            guided = list(set(guided + list(range(rank))))
            for c1, c2 in itertools.combinations(guided, 2):
                if time.time() > deadline - 5:
                    break
                m2 = np.ones(rank, dtype=bool)
                m2[c1] = False
                m2[c2] = False
                U_new = U_np[:, m2]
                V_new = V_np[:, m2]
                W_new = W_np[:, m2]
                res = np.sum((full_decomp(U_new, V_new, W_new) - T)**2)
                if res < 1e-8:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = int(m2.sum())
                    best_residual = res
                    break
    except Exception:
        pass

    if best_rank < rank and best_residual < 1e-8:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U, "V": V, "W": W}
