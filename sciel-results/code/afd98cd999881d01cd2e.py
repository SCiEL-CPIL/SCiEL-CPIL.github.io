import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 6

    # Strassen 2x2 * 2x2 terms: (U_factors, V_factors, W_factors)
    # Each factor is a list of (local_index, coeff).
    # local indices: U: 0=P00,1=P01,2=P10,3=P11 ; V: 0=Q00,1=Q10,2=Q01,3=Q11 ; W: 0=R00,1=R10,2=R01,3=R11
    strassen_terms = [
        ([0, 3], [0, 3], [0, 3]),        # M1=(P00+P11)(Q00+Q11) -> R00+R11
        ([2, 3], [0], [1, 3]),            # M2=(P10+P11)Q00 -> R10-R11
        ([0], [2, 3]),                    # M3=P00(Q01-Q11) -> R01+R11
        ([3], [1, 0]),                    # M4=P11(Q10-Q00) -> R00+R10
        ([0, 1], [3], [0, 2]),            # M5=(P00+P01)Q11 -> -R00+R01
        ([2, 0], [0, 2], [3]),            # M6=(P10-P00)(Q00+Q01) -> R11
        ([1, 3], [1, 3], [0]),            # M7=(P01-P11)(Q10+Q11) -> R00
    ]
    # Per-term coefficient signs (u, v, w)
    term_sgn = [
        (1, 1, 1),
        (1, 1, (1, -1)),
        (1, (1, -1), (1, 1)),
        (1, (1, -1), (1, 1)),
        ((1, 1), 1, (-1, 1)),
        ((1, -1), (1, 1), 1),
        ((1, -1), (1, 1), 1),
    ]

    # 6 blocks: (a_r_off, a_c_off, b_r_off, b_c_off)
    # source=0 (top): A0=A[0:2,0:2], Q=B[0:2, 2c:2c+2]
    # source=1 (bot): A1=A[0:2,2:4], Q=B[2:4, 2c:2c+2]
    blocks = []
    for c in range(3):
        blocks.append((0, 0, 0, 2 * c))     # top
    for c in range(3):
        blocks.append((0, 2, 2, 2 * c))     # bot

    def build_strassen():
        U = np.zeros((n * m, 42))
        V = np.zeros((m * p, 42))
        W = np.zeros((p * n, 42))
        col = 0
        for a_r_off, a_c_off, b_r_off, b_c_off in blocks:
            for t in range(7):
                ufac, vfac, wfac = strassen_terms[t]
                usgn, vsgn, wsgn = term_sgn[t]
                for idx, coeff in ufac:
                    if coeff == 0:
                        continue
                    s = usgn[idx] if isinstance(usgn, tuple) else usgn
                    u_row = (idx // 2 + a_r_off) * m + (idx % 2 + a_c_off)
                    U[u_row, col] += coeff * s
                for idx, coeff in vfac:
                    if coeff == 0:
                        continue
                    s = vsgn[idx] if isinstance(vsgn, tuple) else vsgn
                    v_row = (idx // 2 + b_r_off) * p + (idx % 2 + b_c_off)
                    V[v_row, col] += coeff * s
                for idx, coeff in wfac:
                    if coeff == 0:
                        continue
                    s = wsgn[idx] if isinstance(wsgn, tuple) else wsgn
                    r_i = idx // 2 + a_r_off
                    r_k = idx % 2 + b_c_off
                    W[r_k * n + r_i, col] += coeff * s
                col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1) > tol:
                        return False
        return True

    def round_half(X):
        return np.round(X * 2) / 2

    def rank_of_factors(U, V, W):
        """Number of rank-1 terms (columns) needed = max column rank of U,V,W."""
        rU = np.linalg.matrix_rank(U, tol=1e-9)
        rV = np.linalg.matrix_rank(V, tol=1e-9)
        rW = np.linalg.matrix_rank(W, tol=1e-9)
        return int(max(rU, rV, rW))

    def prune(U, V, W):
        """Keep the first R columns that span each factor; R = max column rank.
        Since the tensor is preserved under any invertible column mix, keeping a
        basis of the column spaces preserves exactness."""
        R = rank_of_factors(U, V, W)
        if R == 0:
            return U, V, W
        # pick columns greedily to maximize rank of each factor simultaneously
        # Simple: keep first R columns; verify; if fails, try more.
        for R_try in range(R, U.shape[1] + 1):
            U2 = U[:, :R_try]
            V2 = V[:, :R_try]
            W2 = W[:, :R_try]
            if rank_of_factors(U2, V2, W2) == R_try and verify(U2, V2, W2):
                return U2, V2, W2
        return U, V, W

    def random_change(U, V, W):
        """Apply a random invertible rank x rank mix to the columns (preserves tensor)."""
        rank = U.shape[1]
        G = np.eye(rank)
        # add a random sparse perturbation
        G = G + 0.5 * np.random.randn(rank, rank)
        if abs(np.linalg.det(G)) < 1e-6:
            G = np.eye(rank)
        U2 = U @ G
        V2 = V @ G
        W2 = W @ np.linalg.inv(G)
        return U2, V2, W2

    # 1) Build the structured rank-42 decomposition
    U, V, W = build_strassen()
    assert verify(U, V, W), "Strassen construction failed verification"

    best = (U, V, W)
    best_rank = rank_of_factors(U, V, W)

    # 2) Prune linearly dependent factors
    U, V, W = prune(U, V, W)
    if verify(U, V, W):
        best = (U, V, W)
        best_rank = rank_of_factors(U, V, W)

    # 3) Try random column mixes to expose more sparsity, then prune + round
    attempts = 0
    while time.time() < deadline - 15 and attempts < 40:
        attempts += 1
        U2, V2, W2 = random_change(best[0], best[1], best[2])
        U2, V2, W2 = prune(U2, V2, W2)
        # round to half-integers and verify
        Ur, Vr, Wr = round_half(U2), round_half(V2), round_half(W2)
        if verify(Ur, Vr, Wr):
            r = rank_of_factors(Ur, Vr, Wr)
            if r < best_rank:
                best_rank = r
                best = (Ur, Vr, Wr)
        else:
            # keep unrounded but pruned if it reduces rank and is exact
            if verify(U2, V2, W2):
                r = rank_of_factors(U2, V2, W2)
                if r < best_rank:
                    best_rank = r
                    best = (U2, V2, W2)

    # 4) Final: round best to half-integers if possible
    Ur, Vr, Wr = round_half(best[0]), round_half(best[1]), round_half(best[2])
    if verify(Ur, Vr, Wr):
        best = (Ur, Vr, Wr)

    U, V, W = best
    U = round_half(U)
    V = round_half(V)
    W = round_half(W)

    # Final safety verify; if rounding broke it, use the verified best
    if not verify(U, V, W):
        U, V, W = best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
