import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Strassen 2x2x2 decomposition (7 terms)
    # Local index: A[i*2+j], B[j*2+k], C[k*2+i]
    strassen_decomp = [
        ({0: 1, 3: 1},   {0: 1, 3: 1},   {0: 1, 3: 1}),
        ({2: 1, 3: 1},   {0: 1},         {2: 1, 3: -1}),
        ({0: 1},         {1: 1, 3: -1},  {1: 1, 3: 1}),
        ({3: 1},         {2: 1, 0: -1},  {0: 1, 2: 1}),
        ({0: 1, 1: 1},   {3: 1},         {1: 1, 0: -1}),
        ({2: 1, 0: -1},  {0: 1, 1: 1},   {3: 1}),
        ({1: 1, 3: -1},  {2: 1, 3: 1},   {0: 1}),
    ]

    def _build_block_strassen():
        """n=3 -> 2+1, m=4 -> 2+2, p=8 -> 4x2. Rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[li_i * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + li_i][col] += cf
                    col += 1
            for k in range(8):
                for j2 in range(2):
                    gj = 2 * j1 + j2
                    U[2 * m + gj][col] = 1
                    V[gj * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_alt_split():
        """n=3 -> 1+2. Top row 0 schoolbook (32); bottom rows 1,2 Strassen (56)."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += cf
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_4x4_blocks():
        """n=3 -> 1+2, m=4 -> 2+2, p=8 -> 2x4. Rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j1 in range(2):
            for l1 in range(2):
                for j2 in range(2):
                    for k2 in range(4):
                        gj = 2 * j1 + j2
                        gk = 4 * l1 + k2
                        U[0 * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + 0][col] = 1
                        col += 1
                for l2 in range(2):
                    for u_c, v_c, w_c in strassen_decomp:
                        for li, cf in u_c.items():
                            li_i, li_j = li // 2, li % 2
                            U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                        for li, cf in v_c.items():
                            li_i, li_j = li // 2, li % 2
                            V[(2 * j1 + li_i) * p + (4 * l1 + 2 * l2 + li_j)][col] += cf
                        for li, cf in w_c.items():
                            li_i, li_j = li // 2, li % 2
                            W[(4 * l1 + 2 * l2 + li_j) * n + (1 + li_i)][col] += cf
                        col += 1
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _tensor_check(Ua, Va, Wa, idx):
        """Check if removing column idx still gives the identity tensor."""
        r = Ua.shape[1]
        if r <= 1:
            return False
        # Compute T_total - T_idx
        T_total = np.zeros((n, m, p))
        for c in range(r):
            T_total += Ua[:, c].reshape(n, m, 1) * Va[:, c].reshape(1, m, p) * Wa[:, c].reshape(p, 1, n)
        T_idx = Ua[:, idx].reshape(n, m, 1) * Va[:, idx].reshape(1, m, p) * Wa[:, idx].reshape(p, 1, n)
        return np.allclose(T_total - T_idx, 1.0, atol=1e-9)

    def _reduce(U, V, W, budget):
        """Multi-pass greedy column removal."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        keep = list(range(r))

        for _pass in range(10):
            if time.time() > budget:
                break
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            r = Ua.shape[1]
            if r == 0:
                break

            # Precompute contributions
            contributions = []
            T_total = np.zeros((n, m, p))
            for c in range(r):
                tc = Ua[:, c].reshape(n, m, 1) * Va[:, c].reshape(1, m, p) * Wa[:, c].reshape(p, 1, n)
                contributions.append(tc)
                T_total += tc

            removed_any = False
            for idx in range(r):
                if time.time() > budget:
                    break
                t_remaining = T_total - contributions[idx]
                if np.allclose(t_remaining, 1.0, atol=1e-9):
                    keep = [c for c in keep if c != idx]
                    removed_any = True
                    break

            if removed_any:
                continue

            # Try pairs
            found_pair = False
            for i in range(r):
                if time.time() > budget:
                    break
                for j in range(i + 1, r):
                    if np.allclose(contributions[i] + contributions[j], 0.0, atol=1e-9):
                        keep = [c for c in keep if c != i and c != j]
                        found_pair = True
                        break
                if found_pair:
                    break
            if found_pair:
                continue

            if not removed_any and not found_pair:
                break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    def _polish(U, V, W, budget):
        """Perturbation-based local search for further column removals."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        r = Ua.shape[1]
        best_rank = r
        best = (U, V, W)

        for trial in range(80):
            if time.time() > budget:
                break
            U2 = Ua.copy()
            V2 = Va.copy()
            W2 = Wa.copy()
            fac = trial % 3
            arr = [U2, V2, W2][fac]
            ii = (trial * 7 + trial * trial) % arr.shape[0]
            jj = (trial * 13 + trial) % arr.shape[1]
            arr[ii, jj] += 0.5

            keep = list(range(r))
            for idx in range(r - 1, -1, -1):
                if time.time() > budget:
                    break
                T_total = np.zeros((n, m, p))
                for c in keep:
                    T_total += U2[:, c].reshape(n, m, 1) * V2[:, c].reshape(1, m, p) * W2[:, c].reshape(p, 1, n)
                T_idx = U2[:, idx].reshape(n, m, 1) * V2[:, idx].reshape(1, m, p) * W2[:, idx].reshape(p, 1, n)
                if not np.allclose(T_total - T_idx, 1.0, atol=1e-9):
                    continue
                keep = [c for c in keep if c != idx]

            if len(keep) < best_rank:
                Uk = [[U2[i][c] for c in keep] for i in range(U2.shape[0])]
                Vk = [[V2[i][c] for c in keep] for i in range(V2.shape[0])]
                Wk = [[W2[i][c] for c in keep] for i in range(W2.shape[0])]
                if _verify(Uk, Vk, Wk):
                    best_rank = len(keep)
                    best = (Uk, Vk, Wk)

        return best[0], best[1], best[2], best_rank

    def _try_mixed_split():
        """Try n=3 as 2+1 but with different m,p splits to find lower rank."""
        # m=4 as 1+1+2, p=8 as 4+4
        # Row 0,1: Strassen on 2x(1+1+2)x(4+4)
        # Row 2: schoolbook
        rank = 0
        U = []
        V = []
        W = []
        # This is complex; skip for now and rely on existing builders
        return None

    best = None
    best_rank = 10 ** 9

    builders = [_build_block_strassen, _build_alt_split, _build_4x4_blocks]
    for builder in builders:
        if time.time() > deadline:
            break
        sol = builder()
        if sol is None:
            continue
        U, V, W = sol["U"], sol["V"], sol["W"]
        if not _verify(U, V, W):
            continue

        current_rank = len(U[0])
        if current_rank < best_rank:
            best = {"U": U, "V": V, "W": W}
            best_rank = current_rank

        for reducer in (_reduce, _polish):
            if time.time() > deadline:
                break
            try:
                U2, V2, W2, rk = reducer(U, V, W, deadline)
                if rk < best_rank and _verify(U2, V2, W2):
                    best = {"U": U2, "V": V2, "W": W2}
                    best_rank = rk
            except Exception:
                pass

    if best is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best
