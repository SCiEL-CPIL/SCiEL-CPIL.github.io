"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0
    R0 = 36
    U = [[0.0] * R0 for _ in range(n * m)]
    V = [[0.0] * R0 for _ in range(m * p)]
    W = [[0.0] * R0 for _ in range(p * n)]

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    idx_list = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    idx_arr = np.array(idx_list, dtype=np.int32)
    u_rows = idx_arr[:, 0]
    v_rows = idx_arr[:, 1]
    w_rows = idx_arr[:, 2]

    def verify_fast(Ua, Va, Wa, R):
        # Vectorized: compute all 40 inner products at once
        prod = (Ua[u_rows, :R] * Va[v_rows, :R] * Wa[w_rows, :R])
        dots = prod.sum(axis=1)
        if not np.all(np.abs(dots - 1.0) < 1e-6):
            return False
        return True

    base_U = U_np.copy()
    base_V = V_np.copy()
    base_W = W_np.copy()

    if not verify_fast(base_U, base_V, base_W, 36):
        return {
            "U": base_U.tolist(),
            "V": base_V.tolist(),
            "W": base_W.tolist()
        }

    best_rank = 36
    best_U = base_U.copy()
    best_V = base_V.copy()
    best_W = base_W.copy()

    def apply_merge(Ua, Va, Wa, c_target, c_source, sgn, R):
        U2 = Ua.copy()
        V2 = Va.copy()
        W2 = Wa.copy()
        U2[:, c_target] += sgn * U2[:, c_source]
        V2[:, c_target] += sgn * V2[:, c_source]
        W2[:, c_target] += sgn * W2[:, c_source]
        mask = np.ones(R, dtype=bool)
        mask[c_source] = False
        return U2[:, mask], V2[:, mask], W2[:, mask]

    l_r_pairs = [(0,7),(1,8),(2,9),(3,10),(4,11),(5,12),(6,13),
                 (14,21),(15,22),(16,23),(17,24),(18,25),(19,26),(20,27)]
    all_pairs = l_r_pairs[:]
    for c1 in range(36):
        for c2 in range(c1+1, 36):
            if (c1, c2) not in l_r_pairs:
                all_pairs.append((c1, c2))

    # Strategy: fix U,V to structured basis, solve W via least-squares + threshold
    # The tensor T has entries T[i,j,k] = 1 if (i,j) in A and (j,k) in B.
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # Fix U,V to the Strassen basis (36 cols), then for each subset of 35 cols,
    # solve for W via least squares and check if it works.
    # Actually, a better approach: use the merge search but with vectorized verify.

    valid_merges_35 = []
    for (c1, c2) in all_pairs:
        if time.time() > deadline - 35:
            break
        for sgn in (1.0, -1.0):
            U35, V35, W35 = apply_merge(base_U, base_V, base_W, c1, c2, sgn, 36)
            if verify_fast(U35, V35, W35, 35):
                valid_merges_35.append((c1, c2, sgn, U35, V35, W35))
                best_rank = 35
                best_U = U35.copy()
                best_V = V35.copy()
                best_W = W35.copy()

    if valid_merges_35 and time.time() < deadline - 20:
        for (c1, c2, sgn, U35, V35, W35) in valid_merges_35:
            if time.time() > deadline - 20:
                break
            for nc1 in range(35):
                if time.time() > deadline - 15:
                    break
                for nc2 in range(nc1+1, 35):
                    if time.time() > deadline - 15:
                        break
                    for sgn2 in (1.0, -1.0):
                        U34, V34, W34 = apply_merge(U35, V35, W35, nc1, nc2, sgn2, 35)
                        if verify_fast(U34, V34, W34, 34):
                            best_rank = 34
                            best_U = U34.copy()
                            best_V = V34.copy()
                            best_W = W34.copy()
                            break
                    if best_rank < 35:
                        break
                if best_rank < 35:
                    break
            if best_rank < 35:
                break

    if best_rank == 35 and valid_merges_35 and time.time() < deadline - 10:
        for (c1, c2, sgn, U35, V35, W35) in valid_merges_35:
            if time.time() > deadline - 10:
                break
            for nc1 in range(35):
                if time.time() > deadline - 8:
                    break
                for nc2 in range(nc1+1, 35):
                    if time.time() > deadline - 8:
                        break
                    for sgn2 in (1.0, -1.0):
                        U34, V34, W34 = apply_merge(U35, V35, W35, nc1, nc2, sgn2, 35)
                        if verify_fast(U34, V34, W34, 34):
                            for mc1 in range(34):
                                if time.time() > deadline - 5:
                                    break
                                for mc2 in range(mc1+1, 34):
                                    if time.time() > deadline - 5:
                                        break
                                    for sgn3 in (1.0, -1.0):
                                        U33, V33, W33 = apply_merge(U34, V34, W34, mc1, mc2, sgn3, 34)
                                        if verify_fast(U33, V33, W33, 33):
                                            best_rank = 33
                                            best_U = U33.copy()
                                            best_V = V33.copy()
                                            best_W = W33.copy()
                                            break
                                    if best_rank < 34:
                                        break
                                if best_rank < 34:
                                    break
                            if best_rank < 34:
                                break
                            break
                    if best_rank < 35:
                        break
                if best_rank < 35:
                    break
            if best_rank < 35:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
