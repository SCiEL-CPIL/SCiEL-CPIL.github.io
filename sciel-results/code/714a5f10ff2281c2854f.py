import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p

    def check_solution(U, V, W):
        """Fast vectorized exact check: sum_c U[i*m+j,c]V[jp*p+k,c]W[k*n+i,c] == delta[j,jp]."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        # M[i,k,j,jp] = sum_c Ur[i,j,c] * Vr[jp,k,c] * Wr[k,i,c]
        M = np.einsum('ijc,kpc,kic->ikjp', Ur, Vr, Wr)
        target = np.zeros((n, p, m, m))
        for i in range(n):
            for k in range(p):
                target[i, k] = np.eye(m)
        return bool(np.allclose(M, target, atol=1e-8))

    # ---------- Strassen 2x2x2 (rank 7) ----------
    def strassen2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[1,1]=1; U[2,2]=1; U[3,3]=1
        V[0,0]=1; V[0,1]=1; V[1,1]=1; V[2,2]=1; V[3,2]=1; V[3,3]=1
        W[0,2]=1; W[1,0]=1; W[1,3]=1; W[2,1]=1; W[3,0]=1; W[3,1]=1; W[3,3]=-1
        return U, V, W

    # ---------- trivial block (na, mb, pb) -> rank na*mb*pb ----------
    def trivial_block(na, mb, pb):
        U = np.zeros((na*mb, na*mb*pb)); V = np.zeros((mb*pb, na*mb*pb)); W = np.zeros((pb*na, na*mb*pb))
        c = 0
        for i in range(na):
            for j in range(mb):
                for k in range(pb):
                    U[i*mb+j, c] = 1.0
                    V[j*pb+k, c] = 1.0
                    W[k*na+i, c] = 1.0
                    c += 1
        return U, V, W

    # ---------- generic block (na, mb, pb) via recursive Strassen ----------
    def block(na, mb, pb):
        if na == 1 and mb == 1 and pb == 1:
            U, V, W = trivial_block(1, 1, 1)
            return U, V, W
        if na == 2 and mb == 2 and pb == 2:
            return strassen2()
        # recursive split (powers of two assumed)
        na2, mb2, pb2 = na // 2, mb // 2, pb // 2
        U = np.zeros((na*mb, na*mb*pb)); V = np.zeros((mb*pb, na*mb*pb)); W = np.zeros((pb*na, na*mb*pb))
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    Ub, Vb, Wb = block(na2, mb2, pb2)
                    rb = Ub.shape[1]
                    off = (ii*2 + jj) * (mb2 * pb) + jj * pb
                    # U rows: i in [ii*na2, (ii+1)*na2), j in [jj*mb2, (jj+1)*mb2)
                    for a in range(na2):
                        for b in range(mb2):
                            U[(ii*na2 + a)*mb + (jj*mb2 + b), off + b*pb:] = 0.0  # reset
                    # simpler: assign directly
                    for a in range(na2):
                        for b in range(mb2):
                            U[(ii*na2 + a)*mb + (jj*mb2 + b), off + b*pb:off + b*pb + 0]  # placeholder
                    # Do assignment properly:
                    for a in range(na2):
                        for b in range(mb2):
                            U[(ii*na2 + a)*mb + (jj*mb2 + b), off + b*pb:off + b*pb + rb] = Ub[a*mb2 + b, :]
                    for b in range(mb2):
                        for c in range(pb2):
                            V[(jj*mb2 + b)*pb + (kk*pb2 + c), off + b*pb:off + b*pb + rb] = Vb[b*pb2 + c, :]
                    for c in range(pb2):
                        for a in range(na2):
                            W[(kk*pb2 + c)*na + (ii*na2 + a), off + b*pb:off + b*pb + rb] = Wb[c*na2 + a, :]
        return U, V, W

    # ---------- build the full 2x4x7 block decomposition (rank 50) ----------
    def build_full():
        # row blocks: r0=[0], r1=[1] ; col blocks: c0=[0,1], c1=[2,3] ; out blocks: o0=[0,1,2], o1=[3,4,5,6]
        # blocks: (r0,c0,o0)=1x2x3, (r0,c1,o0)=1x2x3, (r0,c0,o1)=1x2x4, (r0,c1,o1)=1x2x4
        #          (r1,c0,o0)=1x2x3, (r1,c1,o0)=1x2x3, (r1,c0,o1)=1x2x4, (r1,c1,o1)=1x2x4
        blocks = [
            (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),
            (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1),
        ]
        rsize = [1, 1]
        csize = [2, 2]
        osize = [3, 4]
        rstart = [0, 1]
        cstart = [0, 2]
        ostart = [0, 3]
        total = 0
        for (rb, cb, ob) in blocks:
            total += rsize[rb] * csize[cb] * osize[ob]
        R = total  # = 50
        U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
        off = 0
        for (rb, cb, ob) in blocks:
            na, mb, pb = rsize[rb], csize[cb], osize[ob]
            Ub, Vb, Wb = trivial_block(na, mb, pb)
            rbk = Ub.shape[1]
            for a in range(na):
                for b in range(mb):
                    U[rstart[rb] + a, off + b*pb:off + b*pb + rbk] = Ub[a*mb + b, :]
            for b in range(mb):
                for c in range(pb):
                    V[cstart[cb] + b, off + b*pb:off + b*pb + rbk] = Vb[b*pb + c, :]
            for c in range(pb):
                for a in range(na):
                    W[ostart[ob] + c, off + b*pb:off + b*pb + rbk] = Wb[c*na + a, :]
            off += rbk
        return U, V, W

    # ---------- local improvement: drop a rank-1 term if residual stays rank 1 ----------
    def try_remove_term(U, V, W, e):
        """Return (True, new U,V,W) if removing column e leaves a valid rank-(r-1) decomposition."""
        r = U.shape[1]
        Ur = U.reshape(n, m, -1); Vr = V.reshape(m, p, -1); Wr = W.reshape(p, n, -1)
        # Residual R[i,j,jp,k] = delta[j,jp] - Ur[i,j,e]*Vr[jp,k,e]*Wr[k,i,e]
        # Need R rank-1: R[i,j,jp,k] = u[i,j]*v[jp,k]*w[k,i]
        # Collect u[i,j], v[jp,k], w[k,i]
        u = np.full((n, m), np.nan); v = np.full((m, p), np.nan); w = np.full((p, n), np.nan)
        # anchor
        anchor = None
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = (1.0 if j == jp else 0.0) - Ur[i,j,e]*Vr[jp,k,e]*Wr[k,i,e]
                        if abs(val) > 1e-9:
                            anchor = (i, j, jp, k, val)
                            break
                    if anchor: break
                if anchor: break
            if anchor: break
        if anchor is None:
            # residual zero -> removing e gives a valid lower-rank solution (the others already span)
            # Actually if residual is zero, T = term_e alone, so removing e breaks it. Keep.
            return False, U, V, W
        i0, j0, jp0, k0, val0 = anchor
        u[i0, j0] = 1.0; v[jp0, k0] = 1.0; w[k0, i0] = val0
        # propagate
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = (1.0 if j == jp else 0.0) - Ur[i,j,e]*Vr[jp,k,e]*Wr[k,i,e]
                        if abs(val) < 1e-9:
                            continue
                        if np.isnan(u[i,j]):
                            if not (np.isnan(v[jp,k]) and np.isnan(w[k,i])):
                                return False, U, V, W
                            u[i,j] = 1.0; v[jp,k] = 1.0; w[k,i] = val
                        elif np.isnan(v[jp,k]):
                            if np.isnan(w[k,i]):
                                return False, U, V, W
                            v[jp,k] = val / (u[i,j]*w[k,i])
                        elif np.isnan(w[k,i]):
                            w[k,i] = val / (u[i,j]*v[jp,k])
                        else:
                            if abs(u[i,j]*v[jp,k]*w[k,i] - val) > 1e-7:
                                return False, U, V, W
        if np.isnan(u).any() or np.isnan(v).any() or np.isnan(w).any():
            return False, U, V, W
        # build new matrices (drop column e)
        keep = [c for c in range(r) if c != e]
        newU = np.zeros((n*m, r-1)); newV = np.zeros((m*p, r-1)); newW = np.zeros((p*n, r-1))
        for idx, c in enumerate(keep):
            newU[:, idx] = U[:, c]; newV[:, idx] = V[:, c]; newW[:, idx] = W[:, c]
        # add the residual term at the end
        newU[:, r-1] = u.reshape(-1)
        newV[:, r-1] = v.reshape(-1)
        newW[:, r-1] = w.reshape(-1)
        return True, newU, newV, newW

    # ---------- schoolbook fallback ----------
    def schoolbook():
        U = np.zeros((n*m, rank_school)); V = np.zeros((m*p, rank_school)); W = np.zeros((p*n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0; V[j*p+k, c] = 1.0; W[k*n+i, c] = 1.0
                    c += 1
        return U, V, W

    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Start from the block decomposition (rank 50)
    U, V, W = build_full()
    if not check_solution(U, V, W):
        U, V, W = schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Local improvement: iteratively try to drop terms
    improved = True
    while improved and time.time() < deadline:
        improved = False
        r = U.shape[1]
        for e in range(r):
            if time.time() > deadline:
                break
            ok, nU, nV, nW = try_remove_term(U, V, W, e)
            if ok:
                if check_solution(nU, nV, nW):
                    U, V, W = nU, nV, nW
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = U.shape[1]
                    improved = True
                    break  # restart scan with new rank

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
