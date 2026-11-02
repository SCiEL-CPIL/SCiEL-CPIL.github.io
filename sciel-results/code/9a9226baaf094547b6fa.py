import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
N, M, P = 2, 4, 7
A_SIZE = N * M          # 8
B_SIZE = M * P          # 28
C_SIZE = P * N          # 14
TENSOR_SIZE = N * M * P # 56

def construct():
    start = time.time()
    deadline = start + 118  # safety margin before 150s kill

    # ---------- verification ----------
    def verify(U, V, W, rank):
        U3 = U.reshape(N, M, rank)
        V3 = V.reshape(M, P, rank)
        W3 = W.reshape(P, N, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return bool(np.allclose(T, np.ones((N, M, P)), atol=1e-9))

    def to_int_arrays(U, V, W):
        return (np.array(U, dtype=object), np.array(V, dtype=object), np.array(W, dtype=object))

    # ---------- baseline: block Strassen tiling (rank 50) ----------
    def build_strassen_50():
        rank = 50
        U = np.zeros((A_SIZE, rank), dtype=int)
        V = np.zeros((B_SIZE, rank), dtype=int)
        W = np.zeros((C_SIZE, rank), dtype=int)
        col = 0

        def add_strassen_block(j0, k0):
            nonlocal col
            a_idx = 0 * M + j0
            b_idx = 0 * M + (j0 + 1)
            c_idx = 1 * M + j0
            d_idx = 1 * M + (j0 + 1)
            e_idx = j0 * P + k0
            f_idx = j0 * P + (k0 + 1)
            g_idx = (j0 + 1) * P + k0
            h_idx = (j0 + 1) * P + (k0 + 1)
            c00 = k0 * N + 0
            c01 = (k0 + 1) * N + 0
            c10 = k0 * N + 1
            c11 = (k0 + 1) * N + 1

            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00, col] += 1; W[c11, col] += 1; col += 1

            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10, col] += 1; W[c11, col] -= 1; col += 1

            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01, col] += 1; W[c11, col] += 1; col += 1

            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00, col] += 1; W[c10, col] += 1; col += 1

            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00, col] -= 1; W[c01, col] += 1; col += 1

            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11, col] += 1; col += 1

            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00, col] += 1; col += 1

        def add_schoolbook_221_block(j0, k0):
            nonlocal col
            a_idx = 0 * M + j0
            b_idx = 0 * M + (j0 + 1)
            c_idx = 1 * M + j0
            d_idx = 1 * M + (j0 + 1)
            e_idx = j0 * P + k0
            g_idx = (j0 + 1) * P + k0
            c00 = k0 * N + 0
            c10 = k0 * N + 1

            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10, col] += 1; col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(j_start, k_start)
            add_schoolbook_221_block(j_start, 6)
        return U, V, W, rank

    U0, V0, W0, r0 = build_strassen_50()
    if not verify(U0, V0, W0, r0):
        # schoolbook fallback
        rank_fb = N * M * P
        Uf = np.zeros((A_SIZE, rank_fb), dtype=int)
        Vf = np.zeros((B_SIZE, rank_fb), dtype=int)
        Wf = np.zeros((C_SIZE, rank_fb), dtype=int)
        idx = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    Uf[i * M + j, idx] = 1
                    Vf[j * P + k, idx] = 1
                    Wf[k * N + i, idx] = 1
                    idx += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    best_U, best_V, best_W, best_rank = U0, V0, W0, r0

    # ---------- exact integer "polish": try to reduce rank by finding
    #     dependent rank-1 columns that can be eliminated ----------
    # A column r is "redundant" if its rank-1 contribution can be reproduced
    # by a (rational) linear combination of other columns. If such a relation
    # exists, we can drop column r and adjust others -> lower rank.
    def try_reduce_rank(U, V, W, rank, max_rounds=300):
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        active = list(range(rank))
        for _ in range(max_rounds):
            if time.time() > deadline - 6:
                break
            found = False
            for r in active:
                if len(active) - 1 < 2:
                    break
                others = [a for a in active if a != r]
                # Build matrix: each other column a contributes a rank-1 tensor
                # (U[:,a] outer V[:,a] outer W[:,a]) -> vec; target = vec of column r.
                # Solve least squares for coefficients, check exactness.
                target = (U[:, r] * V[:, r])  # shape (A_SIZE*B_SIZE) but we need full tensor
                # Full tensor vec for column r:
                Ur = U[:, r].reshape(N, M, 1)
                Vr = V[:, r].reshape(M, P, 1)
                Wr = W[:, r].reshape(P, N, 1)
                tr = (Ur * Vr * Wr).reshape(-1)  # (N*M*P)
                # Build coefficient matrix
                cols = []
                for a in others:
                    Ua = U[:, a].reshape(N, M, 1)
                    Va = V[:, a].reshape(M, P, 1)
                    Wa = W[:, a].reshape(P, N, 1)
                    cols.append((Ua * Va * Wa).reshape(-1))
                Mmat = np.column_stack(cols)
                # least squares
                try:
                    coef, _, _, _ = np.linalg.lstsq(Mmat, tr, rcond=None)
                except Exception:
                    continue
                resid = Mmat @ coef - tr
                if np.max(np.abs(resid)) < 1e-7:
                    # apply: U[:,a] += coef[a]*U[:,r], etc., then drop r
                    for idx, a in enumerate(others):
                        U[:, a] += coef[idx] * U[:, r]
                        V[:, a] += coef[idx] * V[:, r]
                        W[:, a] += coef[idx] * W[:, r]
                    active.remove(r)
                    found = True
                    break
            if not found:
                break
        # round to integers
        U = np.rint(U[:, active]).astype(int)
        V = np.rint(V[:, active]).astype(int)
        W = np.rint(W[:, active]).astype(int)
        new_rank = len(active)
        if new_rank < rank and verify(U, V, W, new_rank):
            return U, V, W, new_rank
        return U, V, W, rank

    # Apply polish to baseline
    if time.time() < deadline - 10:
        U1, V1, W1, r1 = try_reduce_rank(U0, V0, W0, r0)
        if r1 < best_rank:
            best_U, best_V, best_W, best_rank = U1, V1, W1, r1

    # ---------- "global" structured search: cross-block factor sharing ----------
    # Build a candidate pool of rank-1 terms and greedily cover the tensor,
    # but allow candidates that mix j-blocks (bypassing the block-diagonal
    # Strassen structure). We then polish.
    if time.time() < deadline - 15:
        # Candidate generation: structured patterns with small support.
        # For each candidate we pick a small set of (i,j), (j,k), (k,i) indices
        # and assign +/-1 so that it contributes a "Strassen-like" pattern.
        # Simpler: generate all rank-1 terms that are "Strassen cells" over
        # any 2x2 sub-block of the 4x7 B and 2x2 of A, plus single terms.
        candidates = []

        # single (schoolbook) terms
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    u = np.zeros(A_SIZE); u[i * M + j] = 1
                    v = np.zeros(B_SIZE); v[j * P + k] = 1
                    w = np.zeros(C_SIZE); w[k * N + i] = 1
                    candidates.append((u, v, w))

        # Strassen 2x2x2 cells over any (i-pair fixed=0,1), j0 in {0,1,2}, k0 in {0..5}
        # (standard 7-term Strassen for a 2x2 A-block x 2x2 B-block)
        def strassen_cells(j0, k0):
            a = 0 * M + j0; b = 0 * M + (j0 + 1)
            c = 1 * M + j0; d = 1 * M + (j0 + 1)
            e = j0 * P + k0; f = j0 * P + (k0 + 1)
            g = (j0 + 1) * P + k0; h = (j0 + 1) * P + (k0 + 1)
            c00 = k0 * N + 0; c01 = (k0 + 1) * N + 0
            c10 = k0 * N + 1; c11 = (k0 + 1) * N + 1
            terms = []
            def term(uv, vv, ww):
                u = np.zeros(A_SIZE); v = np.zeros(B_SIZE); w = np.zeros(C_SIZE)
                for (ix, val) in uv: u[ix] += val
                for (ix, val) in vv: v[ix] += val
                for (ix, val) in ww: w[ix] += val
                terms.append((u, v, w))
            term([(a,1),(d,1)], [(e,1),(h,1)], [(c00,1),(c11,1)])
            term([(c,1),(d,1)], [(e,1)], [(c10,1),(c11,-1)])
            term([(a,1)], [(f,1),(h,-1)], [(c01,1),(c11,1)])
            term([(d,1)], [(g,1),(e,-1)], [(c00,1),(c10,1)])
            term([(a,1),(b,1)], [(h,1)], [(c00,-1),(c01,1)])
            term([(c,1),(a,-1)], [(e,1),(f,1)], [(c11,1)])
            term([(b,1),(d,-1)], [(g,1),(h,1)], [(c00,1)])
            return terms

        for j0 in range(0, M - 1):
            for k0 in range(0, P - 1):
                candidates.extend(strassen_cells(j0, k0))

        # Greedy set-cover with exact residual tracking
        # Represent residual as vector of length TENSOR_SIZE; each candidate
        # has a "coverage" vector (its rank-1 tensor). We need sum of chosen
        # (with integer multipliers) to equal ones. Greedy: repeatedly pick the
        # candidate that best reduces residual in a least-squares sense, but
        # we restrict multipliers to {-1,0,1} to keep integrality, then polish.
        residual = np.ones(TENSOR_SIZE)
        chosen = []
        cand_vecs = []
        for (u, v, w) in candidates:
            vec = (u.reshape(N, M, 1) * v.reshape(M, P, 1) * w.reshape(P, N, 1)).reshape(-1)
            cand_vecs.append(vec)
        cand_vecs = np.array(cand_vecs)
        # dedupe
        seen = {}
        uniq = []
        for idx, vec in enumerate(cand_vecs):
            key = tuple(vec.tolist())
            if key not in seen:
                seen[key] = idx
                uniq.append(idx)
        cand_vecs = cand_vecs[uniq]
        candidates = [candidates[i] for i in uniq]

        # Greedy: while residual not zero, pick candidate minimizing ||residual - c*vec||
        # with c in {-1, 1}
        used = set()
        steps = 0
        while np.max(np.abs(residual)) > 1e-9 and steps < 200:
            if time.time() > deadline - 12:
                break
            best_i = -1
            best_c = 0
            best_err = np.inf
            for i in range(len(cand_vecs)):
                vec = cand_vecs[i]
                for c in (1, -1):
                    err = np.sum((residual - c * vec) ** 2)
                    if err < best_err:
                        best_err = err
                        best_i = i
                        best_c = c
            if best_i < 0 or best_err >= np.sum(residual ** 2) - 1e-9:
                break
            residual -= best_c * cand_vecs[best_i]
            chosen.append((best_i, best_c))
            steps += 1
            if np.max(np.abs(residual)) <= 1e-9:
                break

        if np.max(np.abs(residual)) <= 1e-9 and chosen:
            # assemble
            r = len(chosen)
            U = np.zeros((A_SIZE, r), dtype=int)
            V = np.zeros((B_SIZE, r), dtype=int)
            W = np.zeros((C_SIZE, r), dtype=int)
            for col, (ci, cval) in enumerate(chosen):
                u, v, w = candidates[ci]
                U[:, col] = cval * u
                V[:, col] = cval * v
                W[:, col] = cval * w
            if verify(U, V, W, r):
                if time.time() < deadline - 8:
                    U, V, W, r = try_reduce_rank(U, V, W, r)
                if r < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, r

    # ---------- final polish on best ----------
    if time.time() < deadline - 4 and best_rank > 0:
        U2, V2, W2, r2 = try_reduce_rank(best_U, best_V, best_W, best_rank, max_rounds=500)
        if r2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, r2

    if time.time() > deadline - 2:
        # hard stop
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    valid = np.allclose(T, np.ones((N, M, P)), atol=1e-9)
    print(f"Verification {'Passed' if valid else 'Failed'}. Rank: {rank}")
