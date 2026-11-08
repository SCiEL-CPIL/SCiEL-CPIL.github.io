import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check tensor condition: for all (i,j,k), sum_r U[i,j,r]V[j,k,r]W[k,i,r] == 1."""
    rank = U.shape[1]
    # Build tensors for speed: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    Uf = U.reshape(n, m, rank)
    Vf = V.reshape(m, p, rank)
    Wf = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Uf[i, j, :], Vf[j, k, :] * Wf[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _loss_and_grad(U, V, W, n, m, p):
    """loss = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2 and grads."""
    rank = U.shape[1]
    Uf = U.reshape(n, m, rank)
    Vf = V.reshape(m, p, rank)
    Wf = W.reshape(p, n, rank)
    loss = 0.0
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = Uf[i, j, :] * Vf[j, k, :] * Wf[k, i, :]
                s = prod.sum()
                e = s - 1.0
                loss += e * e
                # d loss / d U[i,j,r] = 2 e * V[j,k,r] * W[k,i,r]
                gU[i * m + j, :] += 2.0 * e * (Vf[j, k, :] * Wf[k, i, :])
                gV[j * p + k, :] += 2.0 * e * (Uf[i, j, :] * Wf[k, i, :])
                gW[k * n + i, :] += 2.0 * e * (Uf[i, j, :] * Vf[j, k, :])
    return loss, gU, gV, gW


def _optimize(U, V, W, n, m, p, max_iter, lr, deadline):
    """L-BFGS-free gradient descent with backtracking. Returns best (U,V,W,loss)."""
    rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())
    best_loss = np.inf
    U, V, W = U.copy(), V.copy(), W.copy()
    for it in range(max_iter):
        if time.time() > deadline:
            break
        loss, gU, gV, gW = _loss_and_grad(U, V, W, n, m, p)
        if loss < best_loss:
            best_loss = loss
            best = (U.copy(), V.copy(), W.copy())
        # adaptive step
        step = lr
        for _ in range(8):
            nU = U - step * gU
            nV = V - step * gV
            nW = W - step * gW
            nl, _, _, _ = _loss_and_grad(nU, nV, nW, n, m, p)
            if nl < loss:
                U, V, W = nU, nV, nW
                if loss < best_loss:
                    best_loss = loss
                    best = (U.copy(), V.copy(), W.copy())
                break
            step *= 0.5
        if step < 1e-12:
            break
    return best, best_loss


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Start from schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = U0.shape[1]

    # Try to reduce rank using a structured ansatz + optimization.
    # Ansatz: use the fact that for fixed (i,k), the j-sum is an inner product.
    # We build a candidate decomposition of rank r by initializing from a
    # "block" structure and polishing with gradient descent.
    for try_rank in range(best_rank - 1, max(best_rank - 12, 18), -1):
        if time.time() > deadline:
            break
        found = False
        for seed in range(4):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(1000 + seed)
            # Initialize by perturbing a subset of schoolbook terms and adding
            # random structure to allow merging.
            U = np.zeros((n * m, try_rank), dtype=float)
            V = np.zeros((m * p, try_rank), dtype=float)
            W = np.zeros((p * n, try_rank), dtype=float)
            # Fill first try_rank schoolbook columns, then randomize to allow recombination
            for r in range(try_rank):
                # map r to a (i,j,k) triple, wrap
                ii = r % n
                jj = (r // n) % m
                kk = (r // (n * m)) % p
                U[ii * m + jj, r] = 1.0
                V[jj * p + kk, r] = 1.0
                W[kk * n + ii, r] = 1.0
            # add small random perturbation to enable movement off the manifold
            U += rng.randn(n * m, try_rank) * 0.05
            V += rng.randn(m * p, try_rank) * 0.05
            W += rng.randn(p * n, try_rank) * 0.05
            # optimize
            cand, loss = _optimize(U, V, W, n, m, p, max_iter=200, lr=0.02, deadline=deadline)
            if loss < 1e-8:
                if _verify(cand[0], cand[1], cand[2], n, m, p, tol=1e-9):
                    best_U, best_V, best_W = cand[0], cand[1], cand[2]
                    best_rank = try_rank
                    found = True
                    break
        if found:
            # try to go lower immediately
            continue

    # Final polish on best solution found (high-precision local solve)
    if time.time() < deadline - 10:
        polished, pl = _optimize(best_U, best_V, best_W, n, m, p,
                                 max_iter=300, lr=0.01, deadline=deadline)
        if pl < 1e-12 and _verify(polished[0], polished[1], polished[2], n, m, p, tol=1e-10):
            best_U, best_V, best_W = polished[0], polished[1], polished[2]

    # Convert to lists with integer/half-integer entries where possible
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                ix = int(round(x * 2))
                # snap to half-integer if very close
                if abs(x - ix / 2.0) < 1e-7:
                    val = ix / 2.0
                    r.append(int(val) if val == int(val) else float(val))
                else:
                    r.append(float(x))
            out.append(r)
        return out

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
