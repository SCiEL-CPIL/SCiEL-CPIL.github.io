import time
import numpy as np

_DIM = (6, 6, 6)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _residual_max(U, V, W, n, m, p):
    # R = (W ⊗ U) (V ⊗ V^T)^T ; residual tensor C[i,k,j] = sum_r U[i,j]V[j,k]W[k,i]
    A = np.kron(W, U)          # (p*n, rank)
    B = np.kron(V, V.T)        # (p*n, rank)
    R = A @ B.T                # (p*n, p*n)
    return float(np.max(np.abs(R)))

def construct():
    n, m, p = _DIM
    deadline = time.time() + 130.0

    U, V, W = _schoolbook(n, m, p)
    best = (U.copy(), V.copy(), W.copy())
    best_res = _residual_max(U, V, W, n, m, p)
    best_rank = U.shape[1]

    rng = np.random.default_rng(12345)

    def try_drop(U, V, W, rank, drop_frac, keep):
        # zero a random subset of columns, refit the rest, test exactness
        cols = np.arange(rank)
        perm = rng.permutation(cols)
        n_drop = int(round(rank * drop_frac))
        drop_idx = set(perm[:n_drop].tolist())
        keep_idx = [c for c in cols if c not in drop_idx]
        if len(keep_idx) < max(1, rank // 4):
            return None
        keep_idx = np.array(keep_idx)

        U0 = U.copy(); V0 = V.copy(); W0 = W.copy()
        U0[:, drop_idx] = 0.0
        V0[:, drop_idx] = 0.0
        W0[:, drop_idx] = 0.0
        Uk = U0[:, keep_idx]
        Vk = V0[:, keep_idx]
        Wk = W0[:, keep_idx]

        # target tensor C[i,k,j]
        C = np.zeros((n, p, m))
        for r in range(len(keep_idx)):
            C += Uk[:, r][:, None] * Wk[:, r][None, :] * Vk[:, r][None, :]
        C_flat = C.reshape(-1)

        # unknowns: Uk (nm*R), Vk (mp*R), Wk (pn*R)
        nm, mp, pn = n * m, m * p, p * n
        R = len(keep_idx)
        # build linear system M x = C_flat where x = [Uk; Vk; Wk]
        # term for Uk[i,j] * (Wk[i,k]*Vk[j,k])
        # We'll build M in blocks.
        rows = nm * mp  # number of C entries
        cols = (nm + mp + pn) * R
        M = np.zeros((rows, cols))
        # Precompute for speed: iterate over r
        for r in range(R):
            uk = Uk[:, r]          # (nm,)
            vk = Vk[:, r]          # (mp,)
            wk = Wk[:, r]          # (pn,)
            # contribution to C[i,k,j] from Uk[i,j]: wk[i,k] * vk[j,k]
            # vector over (i,k,j)
            wv = wk.reshape(n, p)
            vv = vk.reshape(m, p)
            # for fixed (i,k): wv[i,k] * vv[:, k]  -> over j
            # We want M[ (i,k,j), (r, uk) ] = wv[i,k]*vv[j,k]
            # Build per r block
            # uk block: index r*nm + (i*m+j)
            # For each (i,k), j in range m:
            for i in range(n):
                for k in range(p):
                    wv_ik = wv[i, k]
                    if wv_ik == 0:
                        continue
                    for j in range(m):
                        cidx = (i * p + k) * m + j
                        val = wv_ik * vv[j, k]
                        M[cidx, r * nm + (i * m + j)] += val
            # vk block: contribution uk[i,j]*wk[i,k]
            for i in range(n):
                for k in range(p):
                    wv_ik = wv[i, k]
                    if wv_ik == 0:
                        continue
                    for j in range(m):
                        cidx = (i * p + k) * m + j
                        val = uk[i * m + j] * wv_ik
                        M[cidx, nm * R + r * mp + (j * p + k)] += val
            # wk block: contribution uk[i,j]*vk[j,k]
            for i in range(n):
                for k in range(p):
                    vv_jk = [vv[j, k] for j in range(m)]
                    for j in range(m):
                        cidx = (i * p + k) * m + j
                        val = uk[i * m + j] * vv_jk[j]
                        M[cidx, (nm + mp) * R + r * pn + (k * n + i)] += val
        # Solve least squares
        try:
            x, *_ = np.linalg.lstsq(M, C_flat, rcond=None)
        except np.linalg.LinAlgError:
            return None
        Uk_new = x[0: nm * R].reshape(nm, R)
        Vk_new = x[nm * R: nm * R + mp * R].reshape(mp, R)
        Wk_new = x[nm * R + mp * R: cols].reshape(pn, R)

        U_new = U0.copy(); V_new = V0.copy(); W_new = W0.copy()
        U_new[:, keep_idx] = Uk_new
        V_new[:, keep_idx] = Vk_new
        W_new[:, keep_idx] = Wk_new
        res = _residual_max(U_new, V_new, W_new, n, m, p)
        if res < 1e-7:
            return (U_new, V_new, W_new, res, len(keep_idx))
        return None

    # Adaptive rank reduction loop
    cur_U, cur_V, cur_W = U, V, W
    cur_rank = cur_U.shape[1]
    drop_frac = 0.05
    stall = 0
    while time.time() < deadline:
        if stall > 30:
            break
        res = _residual_max(cur_U, cur_V, cur_W, n, m, p)
        if res > 1e-7:
            # recover to best
            cur_U, cur_V, cur_W = best
            cur_rank = best_rank
        result = try_drop(cur_U, cur_V, cur_W, cur_rank, drop_frac, None)
        if result is not None:
            cur_U, cur_V, cur_W, res, new_rank = result
            cur_rank = new_rank
            stall = 0
            if new_rank < best_rank:
                best = (cur_U.copy(), cur_V.copy(), cur_W.copy())
                best_rank = new_rank
                best_res = res
            if cur_rank <= 216 and new_rank < 150:
                drop_frac = min(0.25, drop_frac + 0.01)
        else:
            stall += 1
            drop_frac = max(0.02, drop_frac * 0.9)

    # Final: return best
    Uf, Vf, Wf = best
    # Convert to lists of lists (integer/half-integer friendly: round to 1e-9)
    Uo = [[round(float(v), 9) for v in row] for row in Uf]
    Vo = [[round(float(v), 9) for v in row] for row in Vf]
    Wo = [[round(float(v), 9) for v in row] for row in Wf]
    return {"U": Uo, "V": Vo, "W": Wo}
