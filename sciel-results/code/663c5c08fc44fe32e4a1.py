import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    start = time.time()
    deadline = start + 135

    n, m, p = DIMENSIONS
    n_rows = n * m   # 6
    m_rows = m * p   # 27
    p_rows = p * n   # 18
    total = n * m * p  # 54

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True

    def schoolbook(n_sub, m_sub, p_sub):
        r = n_sub * m_sub * p_sub
        U = [[0]*r for _ in range(n_sub*m_sub)]
        V = [[0]*r for _ in range(m_sub*p_sub)]
        W = [[0]*r for _ in range(p_sub*n_sub)]
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i*m_sub+j][col] = 1
                    V[j*p_sub+k][col] = 1
                    W[k*n_sub+i][col] = 1
                    col += 1
        return U, V, W, r

    def strassen222():
        """Known rank-7 decomposition for 2x2x2 (Strassen)."""
        r = 7
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(4)]
        W = [[0]*r for _ in range(4)]
        # U rows: a=00, b=01, c=10, d=11 ; V rows: e=00,f=01,g=10,h=11 ; W: 00,01,10,11
        # term0: (a+d)(e+h)
        U[0][0]+=1; U[3][0]+=1; V[0][0]+=1; V[3][0]+=1
        W[0][0]+=1; W[3][0]+=1; W[3][0]-=1; W[0][0]+=1
        W[0][0]+=1; W[3][0]-=1
        # Let me use the standard clean form below instead.
        return None

    def strassen222():
        r = 7
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(4)]
        W = [[0]*r for _ in range(4)]
        # a=U00,U01; b=U10,U11 -> index: a00=0,a01=1,a10=2,a11=3
        # e=V00,V01,V10,V11 -> e00=0,e01=1,e10=2,e11=3
        # W: 00=0,01=1,10=2,11=3
        # m0=(a11+b00)(e00+e11)
        U[3][0]+=1; U[2][0]+=1; V[0][0]+=1; V[3][0]+=1
        W[0][0]+=1; W[3][0]+=1; W[3][0]-=1
        W[0][0]+=1; W[3][0]-=1
        # m1=(a10+a11)e00
        U[2][1]+=1; U[3][1]+=1; V[0][1]+=1
        W[2][1]+=1; W[3][1]+=1
        # m2=a00(e01-e11)
        U[0][2]+=1; V[1][2]+=1; V[3][2]-=1
        W[0][2]+=1; W[1][2]+=1
        # m3=a11(e10-e00)
        U[3][3]+=1; V[2][3]+=1; V[0][3]-=1
        W[2][3]+=1; W[3][3]+=1
        # m4=(a00+a01)e11
        U[0][4]+=1; U[1][4]+=1; V[3][4]+=1
        W[1][4]+=1; W[3][4]+=1
        # m5=(a10-a00)(e00+e01)
        U[2][5]+=1; U[0][5]-=1; V[0][5]+=1; V[1][5]+=1
        W[2][5]+=1; W[3][5]+=1
        # m6=(a01-a11)(e10+e11)
        U[1][6]+=1; U[3][6]-=1; V[2][6]+=1; V[3][6]+=1
        W[1][6]+=1; W[3][6]+=1
        return U, V, W, r

    # ---- Rational exact rank-reduction search for a (n,m,p) block ----
    def search_block(n_sub, m_sub, p_sub, target_rank, time_budget, seed):
        """Try to find exact rank <= target_rank decomposition of (n_sub,m_sub,p_sub)
        via bilinear Kronecker structure with rational (integer) entries."""
        rng = np.random.RandomState(seed)
        nrm = n_sub*m_sub
        mrm = m_sub*p_sub
        prm = p_sub*n_sub
        # All (i,j,k) entries
        entries = [(i,j,k) for i in range(n_sub) for j in range(m_sub) for k in range(p_sub)]
        E = len(entries)
        # For each entry, list of (u_row, v_row, w_row)
        coord = {e: (i*m_sub+j, j*p_sub+k, k*n_sub+i) for e,(i,j,k) in enumerate(entries)}

        def try_solve(U, V, W, r):
            """Given fixed U,V,W with r columns, check exact; return None if fail."""
            for e,(i,j,k) in enumerate(entries):
                urow, vrow, wrow = coord[e]
                val = 0
                for c in range(r):
                    val += U[urow][c]*V[vrow][c]*W[wrow][c]
                if val != 1:
                    return None
            return True

        # Strategy: build terms one by one. Each term picks a support.
        # Use a constructive approach: pick a random Kronecker term, subtract its
        # contribution from the target, recurse. Represent target as dict entry->Fraction.
        target = {e: Fraction(1) for e in range(E)}

        def term_contrib(u_sup, v_sup, w_sup):
            """contribution vector (dict entry->Fraction) for a term with supports
            (sets of rows) and coefficient 1."""
            contrib = {}
            for u in u_sup:
                i, j = divmod(u, m_sub)
                for v in v_sup:
                    jj, k = divmod(v, p_sub)
                    if jj != j:
                        continue
                    for w in w_sup:
                        kk, ii = divmod(w, n_sub)
                        if kk != k and ii != i:
                            continue
                        # entry (i,j,k)
                        e = (i*m_sub+j)*p_sub + k  # unique index
                        # recompute e index properly
                        e = i*m_sub*p_sub + j*p_sub + k
                        contrib[e] = contrib.get(e, Fraction(0)) + Fraction(1)
            return contrib

        def e_of(i,j,k):
            return i*m_sub*p_sub + j*p_sub + k

        # Recursive: pick a term, subtract, recurse. Prune on remaining max abs.
        best = [None]

        def rec(rem, depth, U, V, W):
            if time.time() > time_budget:
                return
            if best[0] is not None:
                return
            # check if rem is all zero
            nonzero = [e for e in rem if rem[e] != 0]
            if not nonzero:
                best[0] = (U[:], V[:], W[:])
                return
            if depth >= target_rank:
                return
            # pick an entry with nonzero value to cover
            e0 = nonzero[0]
            i, j, k = (e0 // (m_sub*p_sub), (e0 // p_sub) % m_sub, e0 % p_sub)
            # We need a term whose (u,v,w) supports include (i*m+j, j*p+k, k*n+i)
            # with coefficient 1, and reduce the count of nonzero entries.
            u0, v0, w0 = coord[e0]
            # Try small supports containing u0,v0,w0
            # Enumerate small extra rows to create cancellations
            all_u = list(range(nrm))
            all_v = list(range(mrm))
            all_w = list(range(prm))

            # candidate supports: {u0}, {u0, x}
            cand_u = [{u0}, {u0}]
            for x in all_u:
                if x != u0:
                    cand_u.append({u0, x})
            rng.shuffle(cand_u)
            cand_u = cand_u[:8]

            cand_v = [{v0}]
            for x in all_v:
                if x != v0:
                    cand_v.append({v0, x})
            rng.shuffle(cand_v)
            cand_v = cand_v[:6]

            cand_w = [{w0}]
            for x in all_w:
                if x != w0:
                    cand_w.append({w0, x})
            rng.shuffle(cand_w)
            cand_w = cand_w[:6]

            tried = 0
            for us in cand_u:
                if tried > 40:
                    break
                for vs in cand_v:
                    if tried > 40:
                        break
                    for ws in cand_w:
                        tried += 1
                        # coefficient: to make e0 get exactly 1, coefficient = 1 (since
                        # u0 in us, v0 in vs, w0 in ws -> contributes 1*coef). So coef=1.
                        contrib = term_contrib(us, vs, ws)
                        # check e0 contribution is 1
                        if contrib.get(e0, Fraction(0)) != 1:
                            continue
                        # new remaining
                        new_rem = {}
                        ok = True
                        for e in rem:
                            nv = rem[e] - contrib.get(e, Fraction(0))
                            if nv != 0:
                                new_rem[e] = nv
                        # prune: if any |nv| > 1, impossible with integer terms
                        if any(abs(x) > 1 for x in new_rem.values()):
                            continue
                        nU = U + [[0]*1]
                        nV = V + [[0]*1]
                        nW = W + [[0]*1]
                        c = depth
                        for u in us: nU[u][c] = 1
                        for v in vs: nV[v][c] = 1
                        for w in ws: nW[w][c] = 1
                        rec(new_rem, depth+1, nU, nV, nW)
                        if best[0] is not None:
                            return

        rec(target, 0, [], [], [])
        if best[0] is None:
            return None
        U, V, W = best[0]
        return U, V, W, len(U[0]) if U else 0

    # ---- Build full solution from block decompositions ----
    def build_from_blocks(blocks):
        """blocks: list of (n_sub, m_sub, p_sub, U, V, W, r, i0, j0, k0)
        i0,j0,k0 = offset into the full (n,m,p)."""
        total_rank = sum(b[5] for b in blocks)
        U = [[0]*total_rank for _ in range(n_rows)]
        V = [[0]*total_rank for _ in range(m_rows)]
        W = [[0]*total_rank for _ in range(p_rows)]
        col = 0
        for (ns, ms, ps, Ub, Vb, Wb, rb, i0, j0, k0) in blocks:
            for ii in range(ns):
                for jj in range(ms):
                    for c in range(rb):
                        U[(i0+ii)*m + (j0+jj)][col+c] += Ub[ii*ms+jj][c]
            for jj in range(ms):
                for kk in range(ps):
                    for c in range(rb):
                        V[(j0+jj)*p + (k0+kk)][col+c] += Vb[jj*ps+kk][c]
            for kk in range(ps):
                for ii in range(ns):
                    for c in range(rb):
                        W[(k0+kk)*n + (i0+ii)][col+c] += Wb[kk*ns+ii][c]
            col += rb
        return {"U": U, "V": V, "W": W}

    # ---- Fallback: known rank-50 solution (Strassen 2x2x2 x4 + schoolbook) ----
    def rank50_solution():
        rank = 50
        U = [[0]*rank for _ in range(n_rows)]
        V = [[0]*rank for _ in range(m_rows)]
        W = [[0]*rank for _ in range(p_rows)]
        col = 0
        for b in range(4):
            k0 = 2*b; k1 = 2*b+1
            v_e = 0*p+k0; v_f = 0*p+k1; v_g = 1*p+k0; v_h = 1*p+k1
            u_a = 0*m+0; u_b = 0*m+1; u_c = 1*m+0; u_d = 1*m+1
            U[u_a][col]+=1; U[u_d][col]+=1; V[v_e][col]+=1; V[v_h][col]+=1
            U[u_c][col+1]+=1; U[u_d][col+1]+=1; V[v_e][col+1]+=1
            U[u_a][col+2]+=1; V[v_f][col+2]+=1; V[v_h][col+2]-=1
            U[u_d][col+3]+=1; V[v_g][col+3]+=1; V[v_e][col+3]-=1
            U[u_a][col+4]+=1; U[u_b][col+4]+=1; V[v_h][col+4]+=1
            U[u_c][col+5]+=1; U[u_a][col+5]-=1; V[v_e][col+5]+=1; V[v_f][col+5]+=1
            U[u_b][col+6]+=1; U[u_d][col+6]-=1; V[v_g][col+6]+=1; V[v_h][col+6]+=1
            w_00=k0*n+0; w_01=k1*n+0; w_10=k0*n+1; w_11=k1*n+1
            W[w_00][col]+=1; W[w_00][col+3]+=1; W[w_00][col+4]-=1; W[w_00][col+6]+=1
            W[w_01][col+2]+=1; W[w_01][col+4]+=1
            W[w_10][col+1]+=1; W[w_10][col+3]+=1
            W[w_11][col]+=1; W[w_11][col+1]-=1; W[w_11][col+2]+=1; W[w_11][col+5]+=1
            col += 7
        k8 = 8
        v_e = 0*p+k8; v_g = 1*p+k8
        u_a=0*m+0; u_b=0*m+1; u_c=1*m+0; u_d=1*m+1
        U[u_a][col]+=1; V[v_e][col]+=1
        U[u_b][col+1]+=1; V[v_g][col+1]+=1
        U[u_c][col+2]+=1; V[v_e][col+2]+=1
        U[u_d][col+3]+=1; V[v_g][col+3]+=1
        w_08=k8*n+0; w_18=k8*n+1
        W[w_08][col]+=1; W[w_08][col+1]+=1
        W[w_18][col+2]+=1; W[w_18][col+3]+=1
        col += 4
        u_ar0=0*m+2; u_ar1=1*m+2
        for k in range(9):
            v_b = 2*p+k
            U[u_ar0][col]+=1; V[v_b][col]+=1; W[k*n+0][col]+=1
            U[u_ar1][col+1]+=1; V[v_b][col+1]+=1; W[k*n+1][col+1]+=1
            col += 2
        return {"U": U, "V": V, "W": W}

    # ---- Baseline ----
    base = rank50_solution()
    if verify(base["U"], base["V"], base["W"]):
        best_result = base
        best_rank = 50
    else:
        # ultimate fallback: pure schoolbook
        U,V,W,r = schoolbook(n,m,p)
        best_result = {"U":U,"V":V,"W":W}
        best_rank = r

    # ---- Try to improve: find a low-rank (2,3,3) block, combine with schoolbook (2,3,6) ----
    if time.time() < deadline - 20:
        for target in range(9, 12):
            if time.time() > deadline - 15:
                break
            budget = min(deadline - 10, time.time() + 18)
            res = search_block(2, 3, 3, target, budget, seed=100+target)
            if res is not None:
                Ub, Vb, Wb, rb = res
                # verify the block
                ok = True
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            val = 0
                            for c in range(rb):
                                val += Ub[i*3+j][c]*Vb[j*3+k][c]*Wb[k*2+i][c]
                            if val != 1:
                                ok = False
                if not ok:
                    continue
                # build full: block (2,3,3) at k0=0 + schoolbook (2,3,6) at k0=3
                Us, Vs, Ws, rs = schoolbook(2, 3, 6)
                blocks = [
                    (2,3,3, Ub, Vb, Wb, rb, 0,0,0),
                    (2,3,6, Us, Vs, Ws, rs, 0,0,3),
                ]
                cand = build_from_blocks(blocks)
                if verify(cand["U"], cand["V"], cand["W"]):
                    new_rank = rb + rs
                    if new_rank < best_rank:
                        best_result = cand
                        best_rank = new_rank
                break  # found a valid block; no need for larger target

    return best_result
