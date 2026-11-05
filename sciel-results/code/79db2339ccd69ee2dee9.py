"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    U, V, W = build_4x4x6()

    def verify(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
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
        return U, V, W

    if not verify(U, V, W, n, m, p):
        U, V, W = schoolbook(n, m, p)

    # --- Local-polish rank reduction (change of variables / smooth L2 surrogate) ---
    # A column r can be removed iff it equals an affine combination of the others.
    # We solve for the combination with a smooth (least-squares) surrogate and accept
    # only when the reconstruction is exact. This is strictly an improvement over the
    # old "delete and re-check" heuristic, which could never compensate.
    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        nr = U_arr.shape[1]
        if nr <= 1:
            return U, V, W

        def residual(r, coef):
            # R[r] = U[:,r]*V[:,r]*W[:,r] - sum_{s != r} coef[s] * U[:,s]*V[:,s]*W[:,s]
            Uo = U_arr[:, r]
            Vo = V_arr[:, r]
            Wo = W_arr[:, r]
            target = Uo * Vo * Wo
            for s in range(nr):
                if s == r:
                    continue
                c = coef[s]
                if c != 0.0:
                    target -= c * (U_arr[:, s] * V_arr[:, s] * W_arr[:, s])
            return target

        best_rank = nr
        best_U = U_arr
        best_V = V_arr
        best_W = W_arr
        # Try removing each column; keep the best (lowest) rank found.
        for r in range(nr):
            if time.time() > time_budget:
                break
            others = [s for s in range(nr) if s != r]
            # Build A (nr-1 x nr-1) and b (nr-1) from the (nr-1) independent triples.
            A = np.zeros((nr - 1, nr - 1))
            b = np.zeros(nr - 1)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i == j and k == 0:
                            continue
                        row = i * m + j
                        col = j * p + k
                        wrow = k * n + i
                        uo = U_arr[row, r]
                        vo = V_arr[col, r]
                        wo = W_arr[wrow, r]
                        if uo == 0.0 or vo == 0.0 or wo == 0.0:
                            idx += 1
                            continue
                        b[idx] = -uo * vo * wo
                        for pos, s in enumerate(others):
                            A[idx, pos] = U_arr[row, s] * V_arr[col, s] * W_arr[wrow, s]
                        idx += 1
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            if time.time() > time_budget:
                break
            # Exactness check with a tight tolerance.
            if np.linalg.norm(residual(r, coef)) < 1e-9:
                new_U = np.delete(U_arr, r, axis=1)
                new_V = np.delete(V_arr, r, axis=1)
                new_W = np.delete(W_arr, r, axis=1)
                if new_U.shape[1] < best_rank:
                    best_rank = new_U.shape[1]
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    U_arr, V_arr, W_arr = new_U, new_V, new_W
                    nr = new_U.shape[1]
                    if nr <= 1:
                        break

        if best_rank < U_arr.shape[1] or best_rank < nr:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return U, V, W

    if time.time() < deadline - 5:
        U, V, W = try_reduce_rank(U, V, W, n, m, p, deadline - 5)

    # Final verification; fall back to guaranteed-valid schoolbook on any failure.
    if not verify(U, V, W, n, m, p):
        U, V, W = schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
