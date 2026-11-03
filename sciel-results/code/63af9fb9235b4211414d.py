"""Matrix multiplication 3x5x5 rank search with deadline guard + epigraph polish."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 5]

DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - _elapsed()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1
    U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = -1
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    return U, V, W

def _build_block_decomp(n, m, p, n_part, m_part, p_part):
    U_list, V_list, W_list = [], [], []
    n_offsets, m_offsets, p_offsets = [], [], []
    off = 0
    for s in n_part: n_offsets.append(off); off += s
    off = 0
    for s in m_part: m_offsets.append(off); off += s
    off = 0
    for s in p_part: p_offsets.append(off); off += s

    for ni, ns in enumerate(n_part):
        for mi, ms in enumerate(m_part):
            for pi, ps in enumerate(p_part):
                if ns == 2 and ms == 2 and ps == 2:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    U_full = np.zeros((n * m, 7))
                    V_full = np.zeros((m * p, 7))
                    W_full = np.zeros((p * n, 7))
                    for i in range(2):
                        for j in range(2):
                            U_full[(n_offsets[ni] + i) * m + (m_offsets[mi] + j), :] = U_s[i * 2 + j, :]
                    for j in range(2):
                        for k in range(2):
                            V_full[(m_offsets[mi] + j) * p + (p_offsets[pi] + k), :] = V_s[j * 2 + k, :]
                    for k in range(2):
                        for i in range(2):
                            W_full[(p_offsets[pi] + k) * n + (n_offsets[ni] + i), :] = W_s[k * 2 + i, :]
                    U_list.append(U_full); V_list.append(V_full); W_list.append(W_full)
                else:
                    rank = ns * ms * ps
                    U_full = np.zeros((n * m, rank))
                    V_full = np.zeros((m * p, rank))
                    W_full = np.zeros((p * n, rank))
                    c = 0
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U_full[(n_offsets[ni] + i) * m + (m_offsets[mi] + j), c] = 1
                                V_full[(m_offsets[mi] + j) * p + (p_offsets[pi] + k), c] = 1
                                W_full[(p_offsets[pi] + k) * n + (n_offsets[ni] + i), c] = 1
                                c += 1
                    U_list.append(U_full); V_list.append(V_full); W_list.append(W_full)
    if not U_list:
        return None
    U = np.hstack(U_list); V = np.hstack(V_list); W = np.hstack(W_list)
    Ua = U.reshape(n, m, -1); Va = V.reshape(m, p, -1); Wa = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    if np.max(np.abs(T - 1.0)) < 1e-9:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _block_decomposition(n, m, p):
    best_result, best_rank = None, n * m * p
    all_n_parts, all_m_parts, all_p_parts = [(n,)], [(m,)], [(p,)]
    for a in range(1, n):
        if a >= 1 and n - a >= 1:
            all_n_parts.append((a, n - a))
    for a in range(1, m):
        if a >= 1 and m - a >= 1:
            all_m_parts.append((a, m - a))
    for a in range(1, p):
        if a >= 1 and p - a >= 1:
            all_p_parts.append((a, p - a))
    for n_part in all_n_parts:
        for m_part in all_m_parts:
            for p_part in all_p_parts:
                if _time_left() < 5.0:
                    break
                result = _build_block_decomp(n, m, p, n_part, m_part, p_part)
                if result is not None:
                    rank = len(result["U"][0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
    return best_result if best_result is not None else _schoolbook(), best_rank

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0)) < tol

def _compute_error(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0))

def _als_one_step(U, V, W, n, m, p, rank):
    U_new, V_new, W_new = U.copy(), V.copy(), W.copy()
    for i in range(n):
        for j in range(m):
            A_sys = np.zeros((p, rank))
            for k in range(p):
                A_sys[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                U_new[i * m + j, :] = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)[0]
            except:
                pass
    for j in range(m):
        for k in range(p):
            A_sys = np.zeros((n, rank))
            for i in range(n):
                A_sys[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
            try:
                V_new[j * p + k, :] = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
            except:
                pass
    for k in range(p):
        for i in range(n):
            A_sys = np.zeros((m, rank))
            for j in range(m):
                A_sys[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
            try:
                W_new[k * n + i, :] = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)[0]
            except:
                pass
    return U_new, V_new, W_new

def _epigraph_polish(U, V, W, n, m, p, rank, max_iters=40, tol=1e-9):
    """Smooth epigraph polish: minimize max|T-1| via projected gradient with
    a log-sum-exp smooth surrogate, then snap to half-integers and verify."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    def T_mat(Ua, Va, Wa):
        return np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)

    def smooth_loss(Ua, Va, Wa, beta=200.0):
        T = T_mat(Ua, Va, Wa)
        e = T - 1.0
        # log-sum-exp of |e|^2 as smooth max surrogate
        e2 = e * e
        m = np.max(e2)
        if m < 1e-12:
            return 0.0
        s = np.log(np.sum(np.exp(beta * (e2 - m)))) / beta + m
        return s

    def grad_smooth(Ua, Va, Wa, beta=200.0):
        T = T_mat(Ua, Va, Wa)
        e = T - 1.0
        e2 = e * e
        m = np.max(e2)
        if m < 1e-12:
            return np.zeros_like(Ua), np.zeros_like(Va), np.zeros_like(Wa)
        w = np.exp(beta * (e2 - m))
        w = w / np.sum(w)  # softmax weights over entries
        # d loss / d T = 2*beta*e * w / (beta * sum) ... simplify:
        # loss = (1/beta) log sum exp(beta e2) + m
        # d loss/d e2 = w (softmax)
        # d e2/d T = 2 e
        dT = 2.0 * beta * e * w  # scale by beta for gradient magnitude
        # d T[i,j,k] / d U[i,j,c] = V[j,k,c]*W[k,i,c]
        dU = np.zeros_like(Ua)
        dV = np.zeros_like(Va)
        dW = np.zeros_like(Wa)
        for c in range(rank):
            dU[:, :, c] = np.einsum('ijk,jk,ki->ij', dT, Va[:, :, c], Wa[:, :, c])
            dV[:, :, c] = np.einsum('ijk,ij,ki->jk', dT, Ua[:, :, c], Wa[:, :, c])
            dW[:, :, c] = np.einsum('ijk,ij,jk->ki', dT, Ua[:, :, c], Va[:, :, c])
        return dU, dV, dW

    lr = 0.02
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        dU, dV, dW = grad_smooth(Ua, Va, Wa)
        Ua = Ua - lr * dU
        Va = Va - lr * dV
        Wa = Wa - lr * dW
        # Snap to half-integers and measure
        Uh = np.round(Ua * 2) / 2.0
        Vh = np.round(Va * 2) / 2.0
        Wh = np.round(Wa * 2) / 2.0
        err = _compute_error(Uh, Vh, Wh, n, m, p, rank)
        if err < tol:
            return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
    return None

def _try_als(U0, V0, W0, target_rank, max_iters=30, tol=1e-7):
    n, m, p = DIMENSIONS
    orig_rank = len(U0[0])
    Ua = np.array(U0, dtype=float)
    Va = np.array(V0, dtype=float)
    Wa = np.array(W0, dtype=float)
    for attempt in range(3):
        if _time_left() < 2.0:
            return None
        if attempt == 0 and orig_rank >= target_rank:
            U = Ua[:, :target_rank]; V = Va[:, :target_rank]; W = Wa[:, :target_rank]
        else:
            P = np.random.randn(orig_rank, target_rank)
            U = Ua @ P; V = Va @ P; W = Wa @ P
        for it in range(max_iters):
            if _time_left() < 1.0:
                break
            U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
            Uh = np.round(U * 2) / 2.0
            Vh = np.round(V * 2) / 2.0
            Wh = np.round(W * 2) / 2.0
            err = _compute_error(Uh, Vh, Wh, n, m, p, target_rank)
            if err < tol:
                return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
    return None

def _random_search(n, m, p, target_rank, max_trials=5, tol=1e-7):
    for trial in range(max_trials):
        if _time_left() < 5.0:
            break
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        for c in range(target_rank):
            idx_u = np.random.choice(n * m, size=min(3, n * m), replace=False)
            idx_v = np.random.choice(m * p, size=min(3, m * p), replace=False)
            idx_w = np.random.choice(p * n, size=min(3, p * n), replace=False)
            U[idx_u, c] = np.random.choice([-1, 1], size=len(idx_u))
            V[idx_v, c] = np.random.choice([-1, 1], size=len(idx_v))
            W[idx_w, c] = np.random.choice([-1, 1], size=len(idx_w))
        for it in range(25):
            if _time_left() < 2.0:
                break
            U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
            Uh = np.round(U * 2) / 2.0
            Vh = np.round(V * 2) / 2.0
            Wh = np.round(W * 2) / 2.0
            err = _compute_error(Uh, Vh, Wh, n, m, p, target_rank)
            if err < tol:
                return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
    return None

def _try_strassen_block(n, m, p, target_rank, max_iters=20, tol=1e-7):
    block_result, block_rank = _block_decomposition(n, m, p)
    if block_rank >= target_rank:
        return None
    U = np.array(block_result["U"], dtype=float)
    V = np.array(block_result["V"], dtype=float)
    W = np.array(block_result["W"], dtype=float)
    P = np.random.randn(block_rank, target_rank)
    U = U @ P; V = V @ P; W = W @ P
    for it in range(max_iters):
        if _time_left() < 1.0:
            return None
        U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
        Uh = np.round(U * 2) / 2.0
        Vh = np.round(V * 2) / 2.0
        Wh = np.round(W * 2) / 2.0
        err = _compute_error(Uh, Vh, Wh, n, m, p, target_rank)
        if err < tol:
            return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
    return None

def _local_merge_search(U, V, W, n, m, p, max_tries=200, tol=1e-9):
    """Try to reduce rank by 1: remove a column c and see if remaining columns
    can be adjusted (via small ALS) to still satisfy T=1. Use epigraph polish
    on the reduced system."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Try removing each column in random order
    order = list(range(rank))
    np.random.shuffle(order)
    for c in order[:max_tries]:
        if _time_left() < 2.0:
            break
        U2 = np.delete(Ua, c, axis=1)
        V2 = np.delete(Va, c, axis=1)
        W2 = np.delete(Wa, c, axis=1)
        r2 = U2.shape[1]
        # Run a few ALS steps on reduced system
        U3, V3, W3 = U2.copy(), V2.copy(), W2.copy()
        for it in range(8):
            if _time_left() < 1.0:
                break
            U3, V3, W3 = _als_one_step(U3, V3, W3, n, m, p, r2)
        # Snap and check
        Uh = np.round(U3 * 2) / 2.0
        Vh = np.round(V3 * 2) / 2.0
        Wh = np.round(W3 * 2) / 2.0
        err = _compute_error(Uh, Vh, Wh, n, m, p, r2)
        if err < tol:
            return {"U": Uh.tolist(), "V": Vh.tolist(), "W": Wh.tolist()}
        # Try epigraph polish on the reduced system
        polished = _epigraph_polish(U3, V3, W3, n, m, p, r2, max_iters=20, tol=tol)
        if polished is not None:
            return polished
    return None

def construct():
    if _time_left() < 5.0:
        return _schoolbook()

    np.random.seed(42)
    n, m, p = DIMENSIONS

    block_result, block_rank = _block_decomposition(n, m, p)
    best = block_result
    best_rank = block_rank

    if _time_left() < 10.0:
        if _verify(best["U"], best["V"], best["W"], tol=1e-9):
            return best
        return _schoolbook()

    U0 = np.array(best["U"], dtype=float)
    V0 = np.array(best["V"], dtype=float)
    W0 = np.array(best["W"], dtype=float)

    # Try epigraph polish on the current best to see if it's already near-exact
    polished = _epigraph_polish(U0, V0, W0, n, m, p, best_rank, max_iters=30, tol=1e-9)
    if polished is not None:
        best = polished
        best_rank = best_rank  # same rank, but now verified

    # Try local merge to reduce rank by 1
    if _time_left() > 20.0:
        merged = _local_merge_search(best["U"], best["V"], best["W"], n, m, p, max_tries=100, tol=1e-9)
        if merged is not None:
            best = merged
            best_rank = len(merged["U"][0])

    # Try various target ranks below the block rank with ALS
    for target_rank in range(best_rank - 1, max(30, best_rank - 20), -1):
        if _time_left() < 10.0:
            break
        result = _try_als(best["U"], best["V"], best["W"], target_rank, max_iters=20)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], tol=1e-9):
                best = result
                best_rank = target_rank
                # Try to merge further
                if _time_left() > 15.0:
                    merged = _local_merge_search(best["U"], best["V"], best["W"], n, m, p, max_tries=50, tol=1e-9)
                    if merged is not None:
                        best = merged
                        best_rank = len(merged["U"][0])
                continue

        if _time_left() > 15.0:
            result = _random_search(n, m, p, target_rank, max_trials=5)
            if result is not None:
                if _verify(result["U"], result["V"], result["W"], tol=1e-9):
                    best = result
                    best_rank = target_rank
                    continue

    if _verify(best["U"], best["V"], best["W"], tol=1e-9):
        return best
    else:
        return _schoolbook()
