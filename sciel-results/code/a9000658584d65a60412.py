import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 140

    n, m, p = 2, 3, 9

    def _t(a, b, c, n_, m_, p_):
        i, j = divmod(a, m_)
        jl, k = divmod(b, p_)
        kl, il = divmod(c, n_)
        return float(j == jl and k == kl and i == il)

    def _build_T(n_, m_, p_):
        T = np.zeros((n_*m_, m_*p_, p_*n_))
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    T[i*m_+j, j*p_+k, k*n_+i] = 1.0
        return T

    T_full = _build_T(n, m, p)

    # Schoolbook
    U_sb = np.zeros((n*m, n*m*p)); V_sb = np.zeros((m*p, n*m*p)); W_sb = np.zeros((p*n, n*m*p))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, c] = 1; V_sb[j*p+k, c] = 1; W_sb[k*n+i, c] = 1; c += 1
    best = (n*m*p, U_sb.copy(), V_sb.copy(), W_sb.copy())

    # ---- Search for M_{2,3,3} via random init + ALS with restarts, then expand ----
    n_s, m_s, p_s = 2, 3, 3
    T_s = _build_T(n_s, m_s, p_s)
    best_s_rank = n_s*m_s*p_s  # 18
    best_s_U = best_s_V = best_s_W = None

    def _fit_T(n_, m_, p_, T, r, seed, n_iter=400, tol=1e-9):
        U = np.random.RandomState(seed).randn(n_*m_, r)
        V = np.random.RandomState(seed + 1).randn(m_*p_, r)
        W = np.random.RandomState(seed + 2).randn(p_*n_, r)
        # normalize columns
        for c_ in range(r):
            nrm = np.sqrt((U[:,c_]**2).sum() + (V[:,c_]**2).sum() + (V[:,c_]**2).sum())
            if nrm > 1e-12:
                U[:,c_] /= nrm; V[:,c_] /= nrm; W[:,c_] /= nrm
        prev = np.inf
        for it in range(n_iter):
            # U update
            for a in range(n_*m_):
                B = V[:,None,:] * W[None,:,:]
                M = B.reshape(m_*p_*p_*n_, r).T
                tgt = T[a].reshape(-1)
                try:
                    U[a],_,_,_ = np.linalg.lstsq(M, tgt, rcond=None)
                except Exception:
                    pass
            # V update
            for b in range(m_*p_):
                A = U[:,None,:] * W[None,:,:]
                M = A.reshape(n_*m_*p_*n_, r).T
                tgt = T[:,b,:].reshape(-1)
                try:
                    V[b],_,_,_ = np.linalg.lstsq(M, tgt, rcond=None)
                except Exception:
                    pass
            # W update
            for c_ in range(p_*n_):
                A = U[:,None,:] * V[None,:,:]
                M = A.reshape(n_*m_*m_*p_, r).T
                tgt = T[:,:,c_].reshape(-1)
                try:
                    W[c_],_,_,_ = np.linalg.lstsq(M, tgt, rcond=None)
                except Exception:
                    pass
            err = float(np.sum((np.einsum('ar,br,cr->abc',U,V,W) - T)**2))
            if err < tol:
                return U, V, W, err
            if err > prev + 1e-6 and it > 5:
                # divergence guard
                if err > 1e6:
                    return None, None, None, err
            prev = err
            if it % 50 == 49 and time.time() > deadline - 5:
                break
        return U, V, W, err

    # Try ranks in increasing order
    for r in range(10, 19):
        if r >= best_s_rank:
            break
        if time.time() > deadline - 30:
            break
        found = False
        for trial in range(30):
            if time.time() > deadline - 30:
                break
            U, V, W, err = _fit_T(n_s, m_s, p_s, T_s, r, seed=1000 + r*1000 + trial,
                                   n_iter=300 if r < 14 else 500, tol=1e-9)
            if U is not None and err < 1e-8:
                best_s_rank = r
                best_s_U, best_s_V, best_s_W = U, V, W
                found = True
                break
        if found:
            break

    # Expand M_{2,3,3} to M_{2,3,9} by repeating the k-block 3 times
    if best_s_U is not None:
        r_s = best_s_rank
        U_new = np.tile(best_s_U, (1, 3))
        V_new = np.zeros((m*p, 3*r_s))
        W_new = np.zeros((p*n, 3*r_s))
        for blk in range(3):
            for j in range(m_s):
                for k in range(p_s):
                    rn = j*p + blk*p_s + k
                    rs = j*p_s + k
                    V_new[rn, blk*r_s:(blk+1)*r_s] = best_s_V[rs]
            for k in range(p_s):
                for i in range(n_s):
                    rn = (blk*p_s + k)*n + i
                    rs = k*n_s + i
                    W_new[rn, blk*r_s:(blk+1)*r_s] = best_s_W[rs]
        approx = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        err = float(np.sum((approx - T_full)**2))
        if err < 1e-6:
            best = (3*r_s, U_new, V_new, W_new)

    # ---- Optional: direct ALS on full tensor to try to beat block construction ----
    if time.time() < deadline - 60:
        # Try a few ranks below best[0]
        for r in range(max(10, best[0]-2), best[0]):
            if time.time() > deadline - 30:
                break
            for trial in range(5):
                if time.time() > deadline - 30:
                    break
                U, V, W, err = _fit_T(n, m, p, T_full, r, seed=5000 + r*100 + trial,
                                       n_iter=80, tol=1e-9)
                if U is not None and err < 1e-8:
                    best = (r, U, V, W)
                    break
            else:
                continue
            break

    def to_list(X):
        return [[float(x) for x in row] for row in X]

    _, U, V, W = best
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
