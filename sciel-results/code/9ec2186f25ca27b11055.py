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

    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = rank

    cols = list(range(rank))

    # Search for the lowest rank by checking removals of 3, then 2, then 1 columns.
    # We check combinations from the original set.
    
    # Try removing 3 columns (Rank 47)
    for c1, c2, c3 in itertools.combinations(cols, 3):
        if time.time() > deadline - 10:
            break
        mask = np.ones(rank, dtype=bool)
        mask[c1] = False
        mask[c2] = False
        mask[c3] = False
        D_rem = full_decomp(U_np[:, mask], V_np[:, mask], W_np[:, mask])
        if np.allclose(D_rem, T, atol=1e-9):
            best_U, best_V, best_W = U_np[:, mask], V_np[:, mask], W_np[:, mask]
            best_rank = rank - 3
            break

    # Try removing 2 columns (Rank 48)
    if best_rank == rank:
        for c1, c2 in itertools.combinations(cols, 2):
            if time.time() > deadline - 10:
                break
            mask = np.ones(rank, dtype=bool)
            mask[c1] = False
            mask[c2] = False
            D_rem = full_decomp(U_np[:, mask], V_np[:, mask], W_np[:, mask])
            if np.allclose(D_rem, T, atol=1e-9):
                best_U, best_V, best_W = U_np[:, mask], V_np[:, mask], W_np[:, mask]
                best_rank = rank - 2
                break

    # Try removing 1 column (Rank 49)
    if best_rank == rank:
        for c in cols:
            if time.time() > deadline - 10:
                break
            mask = np.ones(rank, dtype=bool)
            mask[c] = False
            D_rem = full_decomp(U_np[:, mask], V_np[:, mask], W_np[:, mask])
            if np.allclose(D_rem, T, atol=1e-9):
                best_U, best_V, best_W = U_np[:, mask], V_np[:, mask], W_np[:, mask]
                best_rank = rank - 1
                break

    # Final verification with tighter tolerance
    if best_rank < rank:
        final_res = np.sum((full_decomp(best_U, best_V, best_W) - T)**2)
        if final_res < 1e-8:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U, "V": V, "W": W}
