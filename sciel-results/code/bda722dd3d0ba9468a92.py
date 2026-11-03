import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    # ---------- schoolbook baseline ----------
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    def verify(U, V, W, atol=1e-9):
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=atol)

    # ---------- exact 7-term Strassen for a 2x2x2 block ----------
    # indices i0<i1, j0<j1, k0<k1
    # returns 7 (u,v,w) triples that exactly span the 2x2x2 multiplication
    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        def mk_u(*pos):
            u = np.zeros(nm)
            for a, b in pos:
                u[a * m + b] = 1.0
            return u
        def mk_v(*pos):
            v = np.zeros(mp)
            for a, b in pos:
                v[a * p + b] = 1.0
            return v
        def mk_w(*pos):
            w = np.zeros(pn)
            for a, b in pos:
                w[a * n + b] = 1.0
            return w
        # M1 = (A00+A11)(B00+B11)
        t1 = (mk_u((i0, j0), (i1, j1)), mk_v((j0, k0), (j1, k1)), mk_w((k0, i0), (k1, i1)))
        # M2 = (A10+A11) B00
        t2 = (mk_u((i1, j0), (i1, j1)), mk_v((j0, k0)), mk_w((k0, i1)))
        # M3 = A00 (B01-B11)
        t3 = (mk_u((i0, j0)), mk_v((j0, k1), (j1, k1)), mk_w((k1, i0)))
        # M4 = A11 (B10-B00)
        t4 = (mk_u((i1, j1)), mk_v((j1, k0), (j0, k0)), mk_w((k0, i1)))
        # M5 = (A00+A01) B11
        t5 = (mk_u((i0, j0), (i0, j1)), mk_v((j1, k1)), mk_w((k1, i0)))
        # M6 = (A10-A00)(B00+B01)
        t6 = (mk_u((i1, j0), (i0, j0)), mk_v((j0, k0), (j0, k1)), mk_w((k1, i1)))
        # M7 = (A01-A11)(B10+B11)
        t7 = (mk_u((i0, j1), (i1, j1)), mk_v((j1, k0), (j1, k1)), mk_w((k0, i0)))
        return [t1, t2, t3, t4, t5, t6, t7]

    # sanity: verify the 7-term identity on a single block in isolation
    def block_is_exact(i0, i1, j0, j1, k0, k1):
        terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
        # build the 8-point sub-tensor
        for (i, j, k) in [(a, b, c) for a in (i0, i1) for b in (j0, j1) for c in (k0, k1)]:
            s = 0.0
            for (u, v, w) in terms:
                s += u[i * m + j] * v[j * p + k] * w[k * n + i]
            if abs(s - 1.0) > 1e-12:
                return False
        return True

    # ---------- candidate blocks ----------
    I_sets = [frozenset({i, i + 1}) for i in range(n - 1)]
    J_sets = [frozenset({j, j + 1}) for j in range(m - 1)]
    K_sets = [frozenset({k, k + 1}) for k in range(p - 1)]

    all_blocks = []
    for I in I_sets:
        for J in J_sets:
            for K in K_sets:
                i0, i1 = sorted(I)
                j0, j1 = sorted(J)
                k0, k1 = sorted(K)
                if not block_is_exact(i0, i1, j0, j1, k0, k1):
                    continue
                all_blocks.append((frozenset(I), frozenset(J), frozenset(K), i0, i1, j0, j1, k0, k1))

    def blocks_disjoint(b1, b2):
        # disjoint as sets of (i,j,k) points: need I1∩I2=∅ OR J1∩J2=∅ OR K1∩K2=∅
        I1, J1, K1 = b1[0], b1[1], b1[2]
        I2, J2, K2 = b2[0], b2[1], b2[2]
        return not (I1 & I2 and J1 & J2 and K1 & K2)

    # ---------- exact max set packing (backtracking, small universe) ----------
    def max_packing(blocks, deadline):
        best = []
        best_set = []
        # order blocks by "scarcity" to prune: blocks that conflict with many others first
        import sys
        sys.setrecursionlimit(10000)

        # precompute conflict bitmasks
        B = len(blocks)
        conflict = [0] * B
        for a in range(B):
            for b in range(a + 1, B):
                if not blocks_disjoint(blocks[a], blocks[b]):
                    conflict[a] |= (1 << b)
                    conflict[b] |= (1 << a)

        def bt(start, chosen, chosen_mask):
            nonlocal best, best_set
            if time.time() > deadline:
                return
            if len(chosen) > len(best):
                best = list(chosen)
                best_set = list(chosen)
            # upper bound: can we beat best?
            # remaining candidates from start..
            # simple bound: len(chosen) + (B - start) <= len(best) -> prune
            if len(chosen) + (B - start) <= len(best):
                return
            for idx in range(start, B):
                if (chosen_mask >> idx) & 1:
                    continue
                # conflicts with already chosen?
                ok = True
                for c in chosen:
                    if (conflict[idx] >> c) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                chosen.append(idx)
                new_mask = chosen_mask | (1 << idx)
                bt(idx + 1, chosen, new_mask)
                chosen.pop()
                if time.time() > deadline:
                    return

        bt(0, [], 0)
        return best_set

    # ---------- build full decomposition from a packing ----------
    def build_decomposition(packing):
        U_list, V_list, W_list = [], [], []
        covered = set()
        for b in packing:
            _, _, _, i0, i1, j0, j1, k0, k1 = b
            terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
            for u, v, w in terms:
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
            for i in (i0, i1):
                for j in (j0, j1):
                    for k in (k0, k1):
                        covered.add((i, j, k))
        # fill uncovered points with schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i * m + j] = 1.0
                        v[j * p + k] = 1.0
                        w[k * n + i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return U, V, W

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # run exact packing search
    pack_deadline = deadline - 30
    packing = max_packing(all_blocks, pack_deadline)
    if packing:
        U, V, W = build_decomposition(packing)
        if verify(U, V, W):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U, V, W

    # ---------- remove redundant / zero-contributing terms ----------
    if best_U is not None and time.time() < deadline - 10:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()

        def remove_redundant(U, V, W):
            removed = True
            while removed and time.time() < deadline - 10:
                removed = False
                R = U.shape[1]
                Ur = U.reshape(n, m, R)
                Vr = V.reshape(m, p, R)
                Wr = W.reshape(p, n, R)
                for c in range(R):
                    if time.time() >= deadline - 10:
                        break
                    ij = set((i, j) for i in range(n) for j in range(m) if Ur[i, j, c] != 0)
                    jk = set((j, k) for j in range(m) for k in range(p) if Vr[j, k, c] != 0)
                    ki = set((k, i) for k in range(p) for i in range(n) if Wr[k, i, c] != 0)
                    if not ij or not jk or not ki:
                        U = np.delete(U, c, axis=1)
                        V = np.delete(V, c, axis=1)
                        W = np.delete(W, c, axis=1)
                        removed = True
                        break
                    # check whether term c contributes to any (i,j,k)
                    contrib = False
                    for (i, j) in ij:
                        for (j2, k) in jk:
                            if j == j2:
                                if (k, i) in ki:
                                    contrib = True
                                    break
                        if contrib:
                            break
                    if not contrib:
                        U = np.delete(U, c, axis=1)
                        V = np.delete(V, c, axis=1)
                        W = np.delete(W, c, axis=1)
                        removed = True
                        break
            return U, V, W

        U, V, W = remove_redundant(U, V, W)
        if verify(U, V, W):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
