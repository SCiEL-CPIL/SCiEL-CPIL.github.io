import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 11]


def _build_strassen_block(U, V, W, r, k0, k1):
    v_e = k0
    v_f = k1
    v_g = 11 + k0
    v_h = 11 + k1
    w_c00 = 2 * k0
    w_c10 = 2 * k0 + 1
    w_c01 = 2 * k1
    w_c11 = 2 * k1 + 1

    U[0, r] += 1; U[3, r] += 1
    V[v_e, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1; W[w_c11, r] += 1
    r += 1
    U[2, r] += 1; U[3, r] += 1
    V[v_e, r] += 1
    W[w_c10, r] += 1; W[w_c11, r] += -1
    r += 1
    U[0, r] += 1
    V[v_f, r] += 1; V[v_h, r] += -1
    W[w_c01, r] += 1; W[w_c11, r] += 1
    r += 1
    U[3, r] += 1
    V[v_g, r] += 1; V[v_e, r] += -1
    W[w_c00, r] += 1; W[w_c10, r] += 1
    r += 1
    U[0, r] += 1; U[1, r] += 1
    V[v_h, r] += 1
    W[w_c00, r] += -1; W[w_c01, r] += 1
    r += 1
    U[2, r] += 1; U[0, r] += -1
    V[v_e, r] += 1; V[v_f, r] += 1
    W[w_c11, r] += 1
    r += 1
    U[1, r] += 1; U[3, r] += -1
    V[v_g, r] += 1; V[v_h, r] += 1
    W[w_c00, r] += 1
    r += 1
    return r


def _build_schoolbook_block(U, V, W, r, k):
    v_e = k
    v_g = 11 + k
    w_c0 = 2 * k
    w_c1 = 2 * k + 1

    U[0, r] += 1; V[v_e, r] += 1; W[w_c0, r] += 1; r += 1
    U[1, r] += 1; V[v_g, r] += 1; W[w_c0, r] += 1; r += 1
    U[2, r] += 1; V[v_e, r] += 1; W[w_c1, r] += 1; r += 1
    U[3, r] += 1; V[v_g, r] += 1; W[w_c1, r] += 1; r += 1
    return r


def _build_rank39():
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    r = 0
    for block in range(5):
        r = _build_strassen_block(U, V, W, r, block*2, block*2 + 1)
    r = _build_schoolbook_block(U, V, W, r, 10)
    return U, V, W


def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T


def _verify(U, V, W, T=None):
    if T is None:
        T = _build_tensor()
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)


def _reconstruct(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _reduce_rank_svd(U, V, W, deadline, T=None):
    if T is None:
        T = _build_tensor()
    T_flat = T.flatten()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    def _valid(Uc, Vc, Wc):
        return np.allclose(np.einsum('ar,br,cr->abc', Uc, Vc, Wc).flatten(), T_flat, atol=1e-9)

    for _ in range(200):
        if time.time() > deadline - 5:
            break
        if best_rank <= 1:
            break
        current_rank = best_U.shape[1]
        removed = False
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            U_red = np.delete(best_U, r, axis=1)
            V_red = np.delete(best_V, r, axis=1)
            W_red = np.delete(best_W, r, axis=1)
            if _valid(U_red, V_red, W_red):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = current_rank - 1
                removed = True
                break
        if not removed:
            break
    if best_rank < U.shape[1]:
        return best_U, best_V, best_W, best_rank
    return None


def _rank1_completion_search(U, V, W, deadline, T=None):
    """
    Fix the first (rank-1) columns, zero out one column, and solve exactly for a
    new rank-1 term that absorbs the residual. If the residual is a valid rank-1
    tensor, we have a (rank-1)-term decomposition.
    """
    if T is None:
        T = _build_tensor()
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    best = None
    best_rank = rank

    # Try zeroing each column r0 and solving for the replacement term.
    for r0 in range(rank):
        if time.time() > deadline - 5:
            break
        # Residual after removing column r0
        U_rem = np.delete(U, r0, axis=1)
        V_rem = np.delete(V, r0, axis=1)
        W_rem = np.delete(W, r0, axis=1)
        recon_rem = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
        residual = T - recon_rem

        # The removed rank-1 term was u0 = U[:,r0], v0 = V[:,r0], w0 = W[:,r0]
        # residual = u0 ⊗ v0 ⊗ w0  (must hold for a rank-1 completion to exist)
        # Check if residual is rank-1: all 2x2 minors of each matricization vanish.
        # Use exact Fraction arithmetic on the residual (which is integer-valued here).
        res = residual
        # Quick numerical rank-1 check first
        # matricize along a: (n*m) x (m*p*p*n)
        A_mat = res.reshape(n*m, -1)
        # Check rank-1 via two minors
        r1_ok = True
        # find two nonzero entries
        nz = np.argwhere(res != 0)
        if len(nz) < 2:
            # residual is zero -> we already have rank-1 removal (handled by greedy)
            if len(nz) == 0:
                newU = U_rem; newV = V_rem; newW = W_rem
                if _verify(newU, newV, newW, T):
                    if newU.shape[1] < best_rank:
                        best_rank = newU.shape[1]
                        best = (newU.copy(), newV.copy(), newW.copy())
            continue

        # Exact rank-1 test using Fraction on a flattened matricization
        try:
            # matricize to (n*m) x (m*p * p*n)
            R = res.reshape(n*m, m*p*p*n)
            # find a pivot
            pivot = None
            for a in range(R.shape[0]):
                for b in range(R.shape[1]):
                    if R[a, b] != 0:
                        pivot = (a, b)
                        break
                if pivot:
                    break
            if pivot is None:
                continue
            a0, b0 = pivot
            # rank 1 iff R[a,b]*R[a0,b0] == R[a,a0? ] ... use: R[a,b] = R[a,b0]*R[a0,b]/R[a0,b0]
            # check all (a,b): R[a,b]*R[a0,b0] == R[a,b0]*R[a0,b]
            frac = False
            for a in range(R.shape[0]):
                if a == a0:
                    continue
                for b in range(R.shape[1]):
                    if b == b0:
                        continue
                    lhs = Fraction(R[a, b]) * Fraction(R[a0, b0])
                    rhs = Fraction(R[a, b0]) * Fraction(R[a0, b])
                    if lhs != rhs:
                        frac = True
                        break
                if frac:
                    break
            if frac:
                continue
            # residual is rank-1 -> the removal is valid (greedy should have caught it)
            newU = U_rem; newV = V_rem; newW = W_rem
            if _verify(newU, newV, newW, T):
                if newU.shape[1] < best_rank:
                    best_rank = newU.shape[1]
                    best = (newU.copy(), newV.copy(), newW.copy())
        except Exception:
            continue

    if best is not None:
        return best[0], best[1], best[2], best_rank
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    U, V, W = _build_rank39()
    T = _build_tensor()

    if not _verify(U, V, W, T):
        # Fallback schoolbook rank 44
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 39

    # SVD/greedy rank reduction
    if time.time() < deadline - 15:
        res = _reduce_rank_svd(U, V, W, deadline, T)
        if res is not None:
            U_r, V_r, W_r, rk = res
            if _verify(U_r, V_r, W_r, T):
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rk

    # Rank-1 completion search on the (possibly reduced) decomposition
    if time.time() < deadline - 5:
        res2 = _rank1_completion_search(best_U, best_V, best_W, deadline, T)
        if res2 is not None:
            U_c, V_c, W_c, rk2 = res2
            if _verify(U_c, V_c, W_c, T) and rk2 < best_rank:
                best_U, best_V, best_W, best_rank = U_c, V_c, W_c, rk2

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
