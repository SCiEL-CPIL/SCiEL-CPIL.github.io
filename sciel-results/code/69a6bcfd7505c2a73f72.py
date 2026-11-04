"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    def strassen_terms():
        T = []
        T.append(([((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1), ((1, 1), 1)]))
        T.append(([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)],
                 [((1, 0), 1), ((1, 1), -1)]))
        T.append(([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)],
                 [((0, 1), 1), ((1, 1), 1)]))
        T.append(([((1, 1), 1)], [((1, 0), 1), ((0, 0), -1)],
                 [((1, 0), 1), ((0, 0), 1)]))
        T.append(([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)],
                 [((0, 0), -1), ((0, 1), 1)]))
        T.append(([((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)],
                 [((1, 1), 1)]))
        T.append(([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1)]))
        return T

    ST = strassen_terms()
    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    def add_rank7(i0, j0, k0, col_start):
        for r in range(7):
            u_e, v_e, w_e = ST[r]
            c = col_start + r
            for (li, lj), val in u_e:
                U[(i0 + li) * m + (j0 + lj), c] += val
            for (lj, lk), val in v_e:
                V[(j0 + lj) * p + (k0 + lk), c] += val
            for (li, lk), val in w_e:
                W[(k0 + lk) * n + (i0 + li), c] += val
        return col_start + 7

    def add_rank4(i0, j0, k0, col_start):
        for t in range(4):
            c = col_start + t
            j = j0 + (t % 2)
            k = k0 + (t // 2)
            U[i0 * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i0, c] = 1
        return col_start + 4

    col = 0
    col = add_rank7(0, 0, 0, col)
    col = add_rank7(0, 0, 2, col)
    col = add_rank7(0, 2, 0, col)
    col = add_rank7(0, 2, 2, col)
    col = add_rank4(2, 0, 0, col)
    col = add_rank4(2, 0, 2, col)
    col = add_rank4(2, 2, 0, col)
    col = add_rank4(2, 2, 2, col)

    assert col == total_rank

    # Attempt rank reduction by checking for linear dependencies
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    removed = []
    for c in range(total_rank - 1, -1, -1):
        if c in removed:
            continue
        # Check if column c is linearly dependent on remaining columns
        # Simple heuristic: check if U[:,c], V[:,c], W[:,c] are all proportional to another
        dependent = False
        for c2 in range(total_rank):
            if c2 == c or c2 in removed:
                continue
            # Check if U[:,c] is proportional to U[:,c2]
            mask_u = U_arr[:, c] != 0
            if np.any(mask_u):
                # Find alpha from non-zero entries
                ratios = U_arr[mask_u, c] / U_arr[mask_u, c2]
                if len(set(np.round(ratios, 10))) == 1:
                    alpha = ratios[0]
                    if np.allclose(U_arr[:, c], alpha * U_arr[:, c2]) and \
                       np.allclose(V_arr[:, c], alpha * V_arr[:, c2]) and \
                       np.allclose(W_arr[:, c], alpha * W_arr[:, c2]):
                        dependent = True
                        break
        
        if dependent:
            removed.append(c)
    
    if removed:
        # Rebuild without removed columns
        keep = [c for c in range(total_rank) if c not in removed]
        U = U_arr[:, keep]
        V = V_arr[:, keep]
        W = W_arr[:, keep]
        total_rank = len(keep)

    # Verify the decomposition
    try:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_comp = np.zeros((n, p))
        for r in range(total_rank):
            ar = sum(U[i * m + j, r] * A[i, j] for i in range(n) for j in range(m))
            br = sum(V[j * p + k, r] * B[j, k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i, r] * ar * br
        max_err = float(np.max(np.abs(C_comp - C_expected)))
        if max_err > 1e-9:
            U_sb, V_sb, W_sb = schoolbook()
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
