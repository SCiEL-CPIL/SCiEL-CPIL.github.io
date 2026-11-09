"""Matrix multiplication 2x3 by 3x7 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Return the multiplication tensor T[i, j, k, l] where l = k*n+i (output index).
    T[i,j,k,l] = 1 iff output (i,k) depends on A[i,j]*B[j,k].
    Here l indexes the output position (k*n + i) per trace convention W[k*n+i, r]."""
    T = np.zeros((n, m, p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = k * n + i
                T[i, j, k, out_idx] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == T[i,j,k,k*n+i]."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Efficient: build via einsum
    Ua = np.asarray(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.asarray(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.asarray(W, dtype=np.float64).reshape(p, n, rank)
    # result[i,j,k,l] = sum_c Ua[i,j,c]*Va[j,k,c]*Wa[k,l,c]
    R = np.einsum('ijc,jkc,klc->ijkl', Ua, Va, Wa)
    diff = np.abs(R - T)
    return bool(np.max(diff) < tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _try_greedy_reduce(U, V, W, n, m, p, deadline):
    """Try to reduce rank by checking if any column is a linear combination of others.
    We check: is there a column c such that U[:,c] is in span of others,
    V[:,c] in span of others, AND W[:,c] in span of others simultaneously?
    If so, we can eliminate it. Repeat until no more reductions."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    tol = 1e-9
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in range(U.shape[1]):
            if time.time() > deadline:
                break
            # Check if column c of U is in span of other columns
            # Similarly for V and W
            ok = True
            for M in (U, V, W):
                others = np.delete(M, c, axis=1)
                col = M[:, c]
                # Check if col is in column space of others
                if others.shape[1] == 0:
                    # only this column; can't eliminate unless zero
                    if np.any(np.abs(col) > tol):
                        ok = False
                        break
                    else:
                        continue
                # least squares
                try:
                    coeff, res, rankM, sv = np.linalg.lstsq(others, col, rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                recon = others @ coeff
                if np.max(np.abs(recon - col)) > tol:
                    ok = False
                    break
            if ok:
                # eliminate column c
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                changed = True
                # verify still exact
                if not _verify(U, V, W, n, m, p):
                    # restore
                    # re-insert
                    U = np.insert(U, c, 0, axis=1)
                    V = np.insert(V, c, 0, axis=1)
                    W = np.insert(W, c, 0, axis=1)
                    changed = False
                else:
                    break  # restart scan
    return U, V, W

def _random_search(rank, n, m, p, deadline, max_iters):
    """Random search over half-integer grid for given rank."""
    best = None
    T = _build_tensor(n, m, p)
    Ua_shape = (n, m, rank)
    Va_shape = (m, p, rank)
    Wa_shape = (p, n, rank)
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=Ua_shape)
        V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=Va_shape)
        W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=Wa_shape)
        R = np.einsum('ijc,jkc,klc->ijkl', U, V, W)
        if np.max(np.abs(R - T)) < 1e-9:
            # found exact!
            return U.reshape(n * m, rank), V.reshape(m * p, rank), W.reshape(p * n, rank)
    return None

def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin: kill at 150s, keep 30s buffer
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_result = None

    # 1. Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_result = (U, V, W)

    # 2. Greedy reduction
    U, V, W = _try_greedy_reduce(U, V, W, n, m, p, deadline)
    if _verify(U, V, W, n, m, p):
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_result = (U, V, W)

    # 3. Random search for lower ranks
    # Try ranks from a reasonable lower bound up to best_rank - 1
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower = max(n * m, m * p, p * n)
    for r in range(lower, best_rank):
        if time.time() > deadline - 10:
            break
        # allocate limited time per rank
        time_per_rank = min(5.0, (deadline - time.time()) / max(1, best_rank - r))
        res = _random_search(r, n, m, p, time.time() + time_per_rank, max_iters=2000)
        if res is not None:
            U, V, W = res
            if _verify(U, V, W, n, m, p):
                best_rank = r
                best_result = (U, V, W)
                break

    # Convert to lists of ints/half-ints
    U, V, W = best_result
    def to_list(M):
        out = []
        for row in M:
            out.append([float(x) if abs(x - round(2*x)) > 1e-9 else (round(2*x)/2) for x in row])
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
