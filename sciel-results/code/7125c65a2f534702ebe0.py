"""Matrix multiplication 4x5x5 rank-minimal decomposition search.

Design family: structured tensor decomposition.
  - We search over a continuous parameterization: a set of rank-1 factors
    plus a compact Strassen-style 4x4x4 block for the top-left sub-problem.
  - Objective: smooth L2 loss  sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
  - Final gate: exact integer/half-integer verification (tol 1e-6).
  - We return the lowest-rank decomposition that verifies; fallback is schoolbook.
"""
import time
import math
import random
import numpy as np

DIMENSIONS = [4, 5, 5]
N, M, P = 4, 5, 5


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n=N, m=M, p=P, tol=1e-6):
    """Verify exact decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1."""
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    r = U_arr.shape[1]
    # Build the tensor efficiently via einsum
    T = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


# ---------------------------------------------------------------------------
# Loss and gradient (smooth L2 surrogate)
# ---------------------------------------------------------------------------
def _loss_and_grad(U, V, W, n=N, m=M, p=P):
    """Return (loss, dU, dV, dW) for the L2 objective."""
    r = U.shape[1]
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # dL/dU[i,j,c] = 2 * T_err[i,j,k] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = T - 1.0
    loss = float(np.sum(err * err))
    dU = 2.0 * np.einsum('ijk,jc,kc->ic', err, V, W)
    dV = 2.0 * np.einsum('ijk,ic,kc->jc', err, U, W)
    dW = 2.0 * np.einsum('ijk,ic,jc->kc', err, U, V)
    return loss, dU, dV, dW


# ---------------------------------------------------------------------------
# Strassen 2x2 factors (for the compact 4x4x4 block)
# ---------------------------------------------------------------------------
def _strassen_2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, -1]
    ], dtype=float)
    return U, V, W


def _strassen_4x4_block():
    """Return (U4, V4, W4) each shape (16, 49) for 4x4x4."""
    U2, V2, W2 = _strassen_2x2()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    for a in range(7):
        for b in range(7):
            c = a * 7 + b
            for i in range(4):
                for j in range(4):
                    flat = i * 4 + j
                    block = (i // 2) * 2 + (j // 2)
                    sub = (i % 2) * 2 + (j % 2)
                    U4[flat, c] = U2[block, a] * U2[sub, b]
                    V4[flat, c] = V2[block, a] * V2[sub, b]
                    W4[flat, c] = W2[block, a] * W2[sub, b]
    return U4, V4, W4


# ---------------------------------------------------------------------------
# Parameterization: rank-r decomposition with a Strassen block + rank-1 tail
# ---------------------------------------------------------------------------
def _build_initial(rank, use_strassen=True, rng=None):
    """Build initial (U, V, W) of shape (20, rank), (25, rank), (20, rank)."""
    if rng is None:
        rng = random.Random(42)
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))

    if use_strassen and rank >= 49:
        U4, V4, W4 = _strassen_4x4_block()
        # Fill the 4x4x4 block (i,j in 0..3, k in 0..3) in the first 49 cols
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    fu = i * M + j
                    fv = j * P + k
                    fw = k * N + i
                    for c in range(49):
                        U[fu, c] += U4[i * 4 + j, c]
                        V[fv, c] += V4[j * 4 + k, c]
                        W[fw, c] += W4[k * 4 + i, c]
        # Rank-1 factors for the remaining 4x4x1 and 4x1x5 parts
        # 4x4x1: i in 0..3, j in 0..3, k=4
        c = 49
        for i in range(4):
            for j in range(4):
                U[i * M + j, c] = 1.0
                V[j * P + 4, c] = 1.0
                W[4 * N + i, c] = 1.0
        # 4x1x5: i in 0..3, j=4, k in 0..4
        c = 50
        for i in range(4):
            for k in range(5):
                U[i * M + 4, c] = 1.0
                V[4 * P + k, c] = 1.0
                W[k * N + i, c] = 1.0
        # If rank > 51, add small random rank-1 terms
        for c in range(51, rank):
            U[:, c] = np.random.randn(N * M) * 0.01
            V[:, c] = np.random.randn(M * P) * 0.01
            W[:, c] = np.random.randn(P * N) * 0.01
    else:
        # Pure rank-1 random initialization
        for c in range(rank):
            U[:, c] = np.random.randn(N * M) * 0.1
            V[:, c] = np.random.randn(M * P) * 0.1
            W[:, c] = np.random.randn(P * N) * 0.1
    return U, V, W


# ---------------------------------------------------------------------------
# Gradient descent with adaptive step
# ---------------------------------------------------------------------------
def _gradient_descent(U, V, W, max_iter, deadline, lr_init=0.01, lr_decay=0.999):
    """Run gradient descent; return best (U,V,W,loss) found."""
    lr = lr_init
    best_loss, best_U, best_V, best_W = float('inf'), U.copy(), V.copy(), W.copy()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        loss, dU, dV, dW = _loss_and_grad(U, V, W)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        # Adaptive step: scale by gradient norm
        gnorm = math.sqrt(float(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2)) + 1e-12)
        step = lr / gnorm
        U -= step * dU
        V -= step * dV
        W -= step * dW
        lr *= lr_decay
    return best_U, best_V, best_W, best_loss


# ---------------------------------------------------------------------------
# Rank-reduction via SVD pruning
# ---------------------------------------------------------------------------
def _try_prune_rank(U, V, W, target_rank, deadline):
    """Try to reduce rank by pruning small columns and re-optimizing."""
    if U.shape[1] <= target_rank:
        return None
    # Compute column norms
    col_norms = np.sqrt(np.sum(U**2, axis=0) * np.sum(V**2, axis=0) * np.sum(W**2, axis=0))
    # Keep the target_rank largest columns
    keep = np.argsort(col_norms)[-target_rank:]
    U2 = U[:, keep].copy()
    V2 = V[:, keep].copy()
    W2 = W[:, keep].copy()
    # Re-optimize
    U2, V2, W2, loss = _gradient_descent(U2, V2, W2, max_iter=200, deadline=deadline, lr_init=0.005)
    return U2, V2, W2, loss


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------
def _search(deadline):
    """Search for the lowest-rank valid decomposition."""
    best = None  # (rank, U, V, W)

    # Try ranks from 49 upward (Strassen-based) and also lower (pure random)
    # We cap total search time.
    time_budget = deadline - time.time()
    if time_budget < 5:
        return None

    # Start with the known rank-51 Strassen construction
    U51, V51, W51 = _build_initial(51, use_strassen=True)
    if _verify(U51, V51, W51):
        best = (51, U51, V51, W51)

    # Try to reduce: optimize rank-51 to see if we can prune
    if time.time() < deadline - 30:
        Uo, Vo, Wo, lo = _gradient_descent(U51, V51, W51, max_iter=300, deadline=deadline - 30, lr_init=0.005)
        if _verify(Uo, Vo, Wo):
            best = (51, Uo, Vo, Wo)
        # Try pruning to 50, 49
        for target in [50, 49]:
            if time.time() > deadline - 20:
                break
            pruned = _try_prune_rank(Uo, Vo, Wo, target, deadline - 10)
            if pruned is not None:
                Up, Vp, Wp, lp = pruned
                if _verify(Up, Vp, Wp):
                    if best is None or target < best[0]:
                        best = (target, Up, Vp, Wp)

    # Try random restarts with lower ranks (pure rank-1)
    if time.time() < deadline - 15:
        for rank in [48, 47, 46, 45, 44, 43, 42, 41, 40]:
            if time.time() > deadline - 10:
                break
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                rng = random.Random(hash((rank, trial)) & 0xFFFFFFFF)
                np.random.seed(hash((rank, trial)) & 0xFFFFFFFF)
                Ur, Vr, Wr = _build_initial(rank, use_strassen=False, rng=rng)
                Ur, Vr, Wr, lr_loss = _gradient_descent(Ur, Vr, Wr, max_iter=200, deadline=deadline - 10, lr_init=0.01)
                if _verify(Ur, Vr, Wr):
                    if best is None or rank < best[0]:
                        best = (rank, Ur, Vr, Wr)

    return best


def _schoolbook_fallback(n=N, m=M, p=P):
    """Trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill

    try:
        result = _search(deadline)
        if result is not None:
            rank, U, V, W = result
            # Final exact verification
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Fallback to schoolbook
        return _schoolbook_fallback()
    except Exception:
        return _schoolbook_fallback()
