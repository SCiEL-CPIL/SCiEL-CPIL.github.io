"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24

    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat = T.reshape(nm * mp, pn)

    start_time = time.time()
    deadline = start_time + 140

    def verify(Ua, Va, Wa):
        # Ua (nm, r), Va (mp, r), Wa (pn, r)
        # sum_c Ua[i*m+j,c]*Va[j*p+k,c]*Wa[k*n+i,c] == 1 for all i,j,k else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k])
                    # s is sum_c Ua[i*m+j,c]*Va[j*p+k,c]; we need sum_c Ua*Va*Wa[k*n+i,c]
                    # recompute properly
                    s = np.einsum('c,c,c->', Ua[i*m+j], Va[j*p+k], Wa[k*n+i])
                    if abs(s - 1.0) > 1e-8:
                        return False
        # check off-diagonal (i,j,k) not matching -> should be 0
        # The tensor is zero unless the triple (i,j,k) is "valid" i.e. always valid for matmul.
        # Actually for matmul, T[i,j,k]=1 for ALL i,j,k (since C[i,k]=sum_j A[i,j]B[j,k], every (i,j,k) contributes).
        # So there are NO off-diagonal zeros to check. Every (i,j,k) in [0..n)x[0..m)x[0..p) has T=1.
        return True

    def verify_full(Ua, Va, Wa):
        """Full check: reconstruct tensor and compare to T."""
        recon = np.zeros((nm, mp, pn))
        r = Ua.shape[1]
        for c in range(r):
            recon += np.einsum('a,b->ab', Ua[:,c], Va[:,c])[:, :, None] * Wa[:,c][None, None, :]
            # that's wrong shape. Do it properly:
        recon = np.zeros((nm, mp, pn))
        for c in range(r):
            recon += np.einsum('a,b,c->abc', Ua[:,c], Va[:,c], Wa[:,c])
        return np.abs(recon - T).max() < 1e-8

    def build_strassen_decomp():
        """Strassen-based rank-68: Strassen on A[0:2,0:2] x B[0:2,b:b+2] for b=0,2,4,6,
        direct for the remaining A rows/cols."""
        U_t, V_t, W_t = [], [], []

        def strassen_decomp(i0, j0, b0):
            # A block rows i0:i0+2, cols j0:j0+2 ; B block rows j0:j0+2, cols b0:b0+2
            # produces C block rows i0:i0+2, cols b0:b0+2
            a = (i0, j0); b = (i0, j0+1); c = (i0+1, j0); d = (i0+1, j0+1)
            e = (j0, b0); f = (j0, b0+1); g = (j0+1, b0); h = (j0+1, b0+1)
            C00 = (i0, b0); C01 = (i0, b0+1); C10 = (i0+1, b0); C11 = (i0+1, b0+1)
            # m1=(a+d)(e+h): C00+=1,C11+=1
            U_t.append([(a,1),(d,1)]); V_t.append([(e,1),(h,1)]); W_t.append([(C00,1),(C11,1)])
            # m2=(c+d)*e: C10+=1,C11-=1
            U_t.append([(c,1),(d,1)]); V_t.append([(e,1)]); W_t.append([(C10,1),(C11,-1)])
            # m3=a*(f-h): C01+=1,C11+=1
            U_t.append([(a,1)]); V_t.append([(f,1),(h,-1)]); W_t.append([(C01,1),(C11,1)])
            # m4=d*(g-e): C00+=1,C10+=1
            U_t.append([(d,1)]); V_t.append([(g,1),(e,-1)]); W_t.append([(C00,1),(C10,1)])
            # m5=(a+b)*h: C00-=1,C01+=1
            U_t.append([(a,1),(b,1)]); V_t.append([(h,1)]); W_t.append([(C00,-1),(C01,1)])
            # m6=(c-a)*(e+f): C11+=1
            U_t.append([(c,1),(a,-1)]); V_t.append([(e,1),(f,1)]); W_t.append([(C11,1)])
            # m7=(b-d)*(g+h): C00+=1
            U_t.append([(b,1),(d,-1)]); V_t.append([(g,1),(h,1)]); W_t.append([(C00,1)])

        def direct(i, j, k):
            U_t.append([((i,j),1)]); V_t.append([((j,k),1)]); W_t.append([((i,k),1)])

        # Strassen on A[0:2,0:2] x B[0:2, b:b+2] for b=0,2,4,6
        for b in range(0, 8, 2):
            strassen_decomp(0, 0, b)
        # A[0:2,2] x B[2, :]
        for i in range(2):
            for k in range(8):
                direct(i, 2, k)
        # A[2, 0:2] x B[0:2, :]
        for j in range(2):
            for k in range(8):
                direct(2, j, k)
        # A[2,2] x B[2, :]
        for k in range(8):
            direct(2, 2, k)

        rank = len(W_t)
        U = np.zeros((nm, rank)); V = np.zeros((mp, rank)); W = np.zeros((pn, rank))
        for idx in range(rank):
            for (i, j), cf in U_t[idx]: U[i*m+j, idx] = cf
            for (j, k), cf in V_t[idx]: V[j*p+k, idx] = cf
            for (i, k), cf in W_t[idx]: W[k*n+i, idx] = cf
        return U, V, W, rank

    U_sb, V_sb, W_sb, sb_rank = build_strassen_decomp()
    if not verify_full(U_sb, V_sb, W_sb):
        # fallback schoolbook
        r = n*m*p
        Uf = np.zeros((nm, r)); Vf = np.zeros((mp, r)); Wf = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j, col] = 1.0; Vf[j*p+k, col] = 1.0; Wf[k*n+i, col] = 1.0; col += 1
        return {"U": [[float(x) for x in row] for row in Uf],
                "V": [[float(x) for x in row] for row in Vf],
                "W": [[float(x) for x in row] for row in Wf]}

    best_rank = sb_rank
    best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())

    def try_rank(r, seed_base, max_attempts, sparsity_opts, vals_opts):
        """Random search for rank r on full 3x3x8. Returns (U,V,W,ok)."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 12:
                return None
            np.random.seed(seed_base + attempt)
            U = np.zeros((nm, r)); V = np.zeros((mp, r))
            sp = np.random.choice(sparsity_opts)
            vals = np.random.choice(vals_opts, size=2)
            for c in range(r):
                uv = np.random.choice(vals, size=nm)
                vv = np.random.choice(vals, size=mp)
                U[:, c] = np.where(np.random.random(nm) < sp, uv, 0.0)
                V[:, c] = np.where(np.random.random(mp) < sp, vv, 0.0)
            K = np.zeros((nm*mp, r))
            for c in range(r):
                K[:, c] = np.outer(U[:, c], V[:, c]).flatten()
            try:
                W_sol, *_ = np.linalg.lstsq(K, T_flat, rcond=1e-12)
            except np.linalg.LinAlgError:
                continue
            if np.abs(K @ W_sol - T_flat).max() < 1e-9:
                # snap to half-integers
                W_snap = np.round(W_sol * 2) / 2
                if verify_full(U, V, W_snap):
                    return (U, V, W_snap, True)
        return None

    def embed_3x3x2(U2, V2, W2, r2):
        """Embed 3x3x2 (n2=m2=3,p2=2) into 3x3x8 by tiling over k-blocks of size 2."""
        n2, m2, p2 = 3, 3, 2
        rank8 = 4 * r2
        U8 = np.zeros((nm, rank8)); V8 = np.zeros((mp, rank8)); W8 = np.zeros((pn, rank8))
        for b in range(4):
            off = b * r2
            U8[:, off:off+r2] = U2
            for j in range(m):
                for k in range(p2):
                    V8[j*p + 2*b + k, off:off+r2] = V2[j*p2 + k, :]
            for k in range(p2):
                for i in range(n):
                    W8[(2*b + k)*n + i, off:off+r2] = W2[k*n2 + i, :]
        return U8, V8, W8, rank8

    def embed_3x3x4(U4, V4, W4, r4):
        n4, m4, p4 = 3, 3, 4
        rank8 = 2 * r4
        U8 = np.zeros((nm, rank8)); V8 = np.zeros((mp, rank8)); W8 = np.zeros((pn, rank8))
        for c in range(r4):
            for half in range(2):
                col = 2*c + half
                U8[:, col] = U4[:, c]
                for j in range(m):
                    for b in range(p4):
                        V8[j*p + 2*b + half, col] = V4[j*p4 + b, c]
                for b in range(p4):
                    for i in range(n):
                        W8[(2*b + half)*n + i, col] = W4[b*n4 + i, c]
        return U8, V8, W8, rank8

    def search_sub(n_s, m_s, p_s, r_lo, r_hi, seed_base, max_attempts, sparsity_opts, vals_opts):
        """Search rank in [r_lo, r_hi] for n_s x m_s x p_s. Returns (U,V,W,r) or None."""
        nm_s = n_s*m_s; mp_s = m_s*p_s; pn_s = p_s*n_s
        Ts = np.zeros((nm_s, mp_s, pn_s))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    Ts[i*m_s+j, j*p_s+k, k*n_s+i] = 1.0
        Ts_flat = Ts.reshape(nm_s*mp_s, pn_s)
        for r in range(r_hi, r_lo-1, -1):
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    return None
                np.random.seed(seed_base + r*1000 + attempt)
                U = np.zeros((nm_s, r)); V = np.zeros((mp_s, r))
                sp = np.random.choice(sparsity_opts)
                vals = np.random.choice(vals_opts, size=2)
                for c in range(r):
                    uv = np.random.choice(vals, size=nm_s)
                    vv = np.random.choice(vals, size=mp_s)
                    U[:, c] = np.where(np.random.random(nm_s) < sp, uv, 0.0)
                    V[:, c] = np.where(np.random.random(mp_s) < sp, vv, 0.0)
                K = np.zeros((nm_s*mp_s, r))
                for c in range(r):
                    K[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                try:
                    W_sol, *_ = np.linalg.lstsq(K, Ts_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                if np.abs(K @ W_sol - Ts_flat).max() < 1e-9:
                    W_snap = np.round(W_sol * 2) / 2
                    recon = np.zeros((nm_s, mp_s, pn_s))
                    for c in range(r):
                        recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W_snap[:,c])
                    if np.abs(recon - Ts).max() < 1e-8:
                        return (U, V, W_snap, r)
        return None

    sp_opts = [0.3, 0.4, 0.5, 0.6]
    val_opts = ([-1.0, 1.0], [-1.0, -0.5, 0.5, 1.0], [-1.0, 1.0, 0.5])

    # 1) Search 3x3x4, embed to 3x3x8 (rank 2*r4)
    if time.time() < deadline - 40:
        res4 = search_sub(3, 3, 4, 15, 24, 70000, 40, sp_opts, val_opts)
        if res4 is not None:
            U4, V4, W4, r4 = res4
            U8, V8, W8, rk = embed_3x3x4(U4, V4, W4, r4)
            if verify_full(U8, V8, W8) and rk < best_rank:
                best_rank = rk
                best_decomp = (U8.copy(), V8.copy(), W8.copy())

    # 2) Search 3x3x2, embed to 3x3x8 (rank 4*r2)
    if best_rank >= 50 and time.time() < deadline - 35:
        res2 = search_sub(3, 3, 2, 10, 16, 30000, 60, sp_opts, val_opts)
        if res2 is not None:
            U2, V2, W2, r2 = res2
            U8, V8, W8, rk = embed_3x3x2(U2, V2, W2, r2)
            if verify_full(U8, V8, W8) and rk < best_rank:
                best_rank = rk
                best_decomp = (U8.copy(), V8.copy(), W8.copy())

    # 3) Direct search on 3x3x8 for ranks below best
    for target in range(best_rank - 1, max(20, best_rank - 12), -1):
        if time.time() > deadline - 20:
            break
        res = try_rank(target, 100000 + target*7, 6, sp_opts, val_opts)
        if res is not None:
            U, V, W, ok = res
            if ok and target < best_rank:
                best_rank = target
                best_decomp = (U.copy(), V.copy(), W.copy())
            break

    U_out, V_out, W_out = best_decomp
    return {"U": [[float(x) for x in row] for row in U_out],
            "V": [[float(x) for x in row] for row in V_out],
            "W": [[float(x) for x in row] for row in W_out]}
