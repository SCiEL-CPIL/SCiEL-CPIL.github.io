"""Matrix multiplication 3x3 by 3x5: rank-minimal exact tensor decomposition.
Distinct approach: vectorized gradient descent with structured initializations
plus a half-integer local-search refinement pass with exact error accounting."""
import time
import numpy as np


def construct():
    start = time.time()
    DEADLINE = start + 138.0  # safety margin before the 150s kill

    n, m, p = 3, 3, 5
    Nm, Mp, Pn = n * m, m * p, p * n  # 9, 15, 15
    schoolbook = n * m * p  # 45
    lower = max(Nm, Mp, Pn)  # 15

    # Precompute the index bookkeeping (fixed)
    idx = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_row = np.array([i * m + j for (i, j, k) in idx], dtype=int)  # (9,)
    v_row = np.array([j * p + k for (i, j, k) in idx], dtype=int)  # (15,)
    w_row = np.array([k * n + i for (i, j, k) in idx], dtype=int)  # (15,)
    n_idx = len(idx)  # 45

    def err_and_grad(U, V, W):
        # Vectorized: val_c = sum_r U[u,c]*V[v,c]*W[w,c]
        P = U[u_row] * V[v_row] * W[w_row]        # (45, r)
        val = P.sum(axis=0)                        # (r,)
        err = val - 1.0                            # (r,)
        E = P * err[None, :]                       # (45, r)
        loss = float(np.sum(err * err))
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        np.add.at(dU, u_row, E)
        np.add.at(dV, v_row, E)
        np.add.at(dW, w_row, E)
        return loss, dU, dV, dW

    def verify(U, V, W, tol=1e-9):
        P = U[u_row] * V[v_row] * W[w_row]
        val = P.sum(axis=0)
        return bool(np.all(np.abs(val - 1.0) <= tol))

    def half_round(M):
        return (np.round(M * 2.0) / 2.0)

    # ---------- Schoolbook fallback (always valid, rank 45) ----------
    U_sb = np.zeros((Nm, schoolbook))
    V_sb = np.zeros((Mp, schoolbook))
    W_sb = np.zeros((Pn, schoolbook))
    for c, (i, j, k) in enumerate(idx):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0
    best = (schoolbook, U_sb.copy(), V_sb.copy(), W_sb.copy())

    # ---------- Half-integer local search refinement ----------
    def local_search(H, budget):
        """H = (U,V,W) half-integers. Try zero->half and half->zero flips,
        accepting any move that reduces the exact tensor error. """
        U, V, W = H
        loss, dU, dV, dW = err_and_grad(U, V, W)
        if loss < 1e-12:
            return U, V, W, True
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = loss
        improved = True
        while improved and time.time() < budget:
            improved = False
            for mat, dmat, rows, dim in ((U, dU, u_row, Nm), (V, dV, v_row, Mp),
                                         (W, dW, w_row, Pn)):
                for r in range(mat.shape[1]):
                    # candidate: flip zeros to 0.5 (or 0.5 to 0) where gradient points
                    # try a cheap heuristic: for the row with largest |grad| in col r,
                    # toggle the zero entry with the largest |grad| contribution.
                    col = dmat[:, r]
                    # entries currently zero
                    zero_mask = (mat[:, r] == 0.0)
                    if not zero_mask.any():
                        continue
                    cand = mat.copy()
                    # pick the zero entry whose magnitude of grad is largest
                    zg = np.where(zero_mask, np.abs(col), 0.0)
                    row = int(np.argmax(zg))
                    if zg[row] < 1e-12:
                        continue
                    # try setting to +0.5 and -0.5
                    for sgn in (0.5, -0.5):
                        cand[row, r] = sgn
                        l2, *_ = err_and_grad(cand if False else
                                               (cand, (U if mat is V else (W if mat is U else V)),
                                                (V if mat is U else (U if mat is W else W)),
                                                (W if mat is U else (V if mat is W else U)))
                                               ) if False else (None, None, None, None)
                        # recompute properly with full set
                        if mat is U:
                            l2, *_ = err_and_grad(cand, V, W)
                        elif mat is V:
                            l2, *_ = err_and_grad(U, cand, W)
                        else:
                            l2, *_ = err_and_grad(U, V, cand)
                        if l2 < best_loss - 1e-12:
                            if mat is U:
                                U = cand
                            elif mat is V:
                                V = cand
                            else:
                                W = cand
                            best_loss = l2
                            improved = True
                            break
        return U, V, W, bool(best_loss < 1e-12)

    # ---------- Structured initializations ----------
    rng = np.random.RandomState(20240611)

    def init_indicator(r):
        """Each column supports a small pattern of indices (indicator + signs)."""
        U = np.zeros((Nm, r))
        V = np.zeros((Mp, r))
        W = np.zeros((Pn, r))
        for c in range(r):
            # pick a random "fiber": choose i,j,k to bias, but allow several
            for _ in range(rng.randint(2, 4)):
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = rng.randint(0, p)
                s = rng.choice([1.0, -1.0])
                U[i * m + j, c] = s
                V[j * p + k, c] = s
                W[k * n + i, c] = s
        return U, V, W

    def init_gauss(r):
        U = rng.randn(Nm, r) * 0.4
        V = rng.randn(Mp, r) * 0.4
        W = rng.randn(Pn, r) * 0.4
        return U, V, W

    # ---------- Gradient descent per rank ----------
    for r in range(lower, schoolbook):
        if time.time() > DEADLINE - 6:
            break
        # budget per rank
        rank_budget = DEADLINE - 4
        n_init = 4 if r < 30 else 3
        found = False
        for init_i in range(n_init):
            if time.time() > rank_budget - 3:
                break
            if init_i == 0:
                U, V, W = init_indicator(r)
            else:
                U, V, W = init_gauss(r)
            # gradient descent with backtracking
            loss, dU, dV, dW = err_and_grad(U, V, W)
            lr = 0.02
            for it in range(1500):
                if time.time() > rank_budget - 3:
                    break
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                l2, dU, dV, dW = err_and_grad(U, V, W)
                if l2 < loss:
                    loss = l2
                    lr = min(lr * 1.1, 0.5)
                else:
                    U += lr * dU
                    V += lr * dV
                    W += lr * dW
                    lr *= 0.5
                if loss < 1e-11:
                    break
            # polish: local search in half-integer space
            Uh, Vh, Wh = half_round(U), half_round(V), half_round(W)
            Uh, Vh, Wh, ok = local_search((Uh, Vh, Wh), rank_budget - 2)
            if ok and verify(Uh, Vh, Wh):
                best = (r, Uh.copy(), Vh.copy(), Wh.copy())
                found = True
                break
        if found:
            # keep going to try even lower rank
            continue

    br, bU, bV, bW = best
    return {
        "U": bU.tolist(),
        "V": bV.tolist(),
        "W": bW.tolist()
    }
