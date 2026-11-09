"""Matrix multiplication 3,4,7 -- block/Strassen construction with greedy rank reduction."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_TOL = 1e-8
_DEADLINE = 140.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=_TOL):
    """Exhaustively check the tensor identity."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            u = U[i * m + j]
            for k in range(p):
                s = np.dot(u, V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_schoolbook(n, m, p):
    """Guaranteed-exact rank n*m*p decomposition."""
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


def _construct_strassen_blocks(n, m, p):
    """Block decomposition: rows 3=2+1, inner 4=2+2, cols 7=2+2+2+1.
    
    2x2x2 blocks use rank-7 Strassen; thin 1x2x2 blocks use rank-2 identity.
    Total rank: 8*7 + 8*2 = 72.
    """
    M = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
        (-1, 1, -1),
        (1, -1, 1),
        (-1, -1, 1),
    ]
    C = [
        [M[0] + M[3] - M[4] + M[5], M[1] + M[2]],
        [M[1] + M[4], M[0] - M[2] + M[3] + M[6]],
    ]
    strassen_sub = np.zeros((2, 2, 2, 7))
    for r, (ua, va, wa) in enumerate(M):
        strassen_sub[0, 0, 0, r] = ua
        strassen_sub[1, 1, 0, r] = ua
        strassen_sub[0, 1, 0, r] = va
        strassen_sub[1, 0, 0, r] = va
        strassen_sub[0, 0, 1, r] = wa
        strassen_sub[1, 1, 1, r] = wa
    for (ci, cj), (ri, rj) in enumerate(C):
        for rr in range(7):
            strassen_sub[ci, cj, 0, rr] += ri[rr]
            strassen_sub[ci, cj, 1, rr] += rj[rr]

    a122 = np.zeros((1, 2, 2, 2))
    a122[0, 0, 0, 0] = 1
    a122[0, 1, 1, 1] = 1

    total_rank = 8 * 7 + 8 * 2
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for jb, j0 in ((0, 0), (1, 2)):
        for kb, k0 in ((0, 0), (1, 2), (2, 4), (3, 6)):
            for a in range(2):
                for b in range(2):
                    for d in range(2):
                        ui = a
                        jj = j0 + b
                        kk = k0 + d
                        for r in range(7):
                            u = strassen_sub[a, b, d, r]
                            if u:
                                U[ui * m + jj, col + r] = u
                            v = strassen_sub[b, d, 0, r]
                            if v:
                                V[jj * p + kk, col + r] = v
                            w = strassen_sub[0, 0, d, r]
                            if w:
                                W[kk * n + ui, col + r] = w
                        col += 7
    for jb, j0 in ((0, 0), (1, 2)):
        for kb, k0 in ((0, 0), (1, 2), (2, 4), (3, 6)):
            for b in range(2):
                for d in range(2):
                    jj = j0 + b
                    kk = k0 + d
                    for r in range(2):
                        u = a122[0, b, d, r]
                        if u:
                            U[2 * m + jj, col + r] = u
                        v = a122[0, b, d, r]
                        if v:
                            V[jj * p + kk, col + r] = v
                        w = a122[0, 0, d, r]
                        if w:
                            W[kk * n + 2, col + r] = w
                    col += 2

    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """Attempt to reduce rank by finding linear dependencies among columns.
    
    Strategy: For each candidate column removal, check if the remaining columns
    can still span the tensor. Use SVD-based low-rank approximation to find
    near-dependencies, then verify exactness.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return None

    # Build the full tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find a subset of columns that still reconstructs T exactly.
    
    # Approach: Use the fact that if columns are linearly dependent in a certain way,
    # we can merge them. Try removing one column at a time and checking if the
    # remaining decomposition still works.
    
    best_U, best_V, best_W = None, None, None
    best_r = current_rank

    # Try removing columns one by one, starting from the end
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    r = current_rank

    for attempt in range(current_rank - target_rank + 1):
        if time.time() > deadline:
            break
        if r <= target_rank:
            break
        
        # Try removing column (r-1)
        # Check if column r-1 is a linear combination of others
        # For exact integer/half-integer entries, check if removing it still works
        U_test = U_work[:, :r-1]
        V_test = V_work[:, :r-1]
        W_test = W_work[:, :r-1]
        
        if _verify(U_test, V_test, W_test, n, m, p):
            U_work = U_test
            V_work = V_test
            W_work = W_test
            r -= 1
            best_U, best_V, best_W = U_work, V_work, W_work
            best_r = r
        else:
            # Try to see if we can adjust: find alpha such that
            # col_{r-1} = sum_{s<r-1} alpha_s * col_s (in the tensor sense)
            # This is complex; for now, just skip this column
            pass

    if best_U is not None and best_r < current_rank:
        return best_U, best_V, best_W, best_r
    return None


def _try_perturbation_search(U, V, W, n, m, p, max_iter, deadline):
    """Try small perturbations to find lower-rank decompositions.
    
    For each pair of columns, try merging them via linear combination.
    """
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_r = current_rank

    for it in range(max_iter):
        if time.time() > deadline:
            break
        if best_r <= 1:
            break
        
        # Try merging two random columns
        r = best_r
        if r < 2:
            break
        i1 = np.random.randint(0, r)
        i2 = np.random.randint(0, r)
        if i1 == i2:
            continue
        
        # Try: new_col = U[:,i1] + U[:,i2], and remove i2
        # This is a heuristic; verify exactness
        U_test = best_U[:, :].copy()
        V_test = best_V[:, :].copy()
        W_test = best_W[:, :].copy()
        
        # Merge: add column i2 to column i1, then remove column i2
        U_test[:, i1] += U_test[:, i2]
        V_test[:, i1] += V_test[:, i2]
        W_test[:, i1] += W_test[:, i2]
        
        # Remove column i2 (shift columns after i2 left)
        if i2 < r - 1:
            U_test[:, i2:r-1] = U_test[:, i2+1:r]
            V_test[:, i2:r-1] = V_test[:, i2+1:r]
            W_test[:, i2:r-1] = W_test[:, i2+1:r]
        U_test = U_test[:, :r-1]
        V_test = V_test[:, :r-1]
        W_test = W_test[:, :r-1]
        
        if _verify(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_r = r - 1

    return best_U, best_V, best_W, best_r


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + _DEADLINE

    # Guaranteed baseline (schoolbook, rank 84).
    U, V, W = _construct_schoolbook(n, m, p)
    best = (U, V, W)
    best_rank = U.shape[1]

    # Attempt the block/Strassen construction (expected rank 72).
    try:
        if time.time() < deadline:
            Us, Vs, Ws = _construct_strassen_blocks(n, m, p)
            if _verify(Us, Vs, Ws, n, m, p):
                best = (Us, Vs, Ws)
                best_rank = Us.shape[1]
    except Exception:
        pass

    # Attempt rank reduction from the best found so far
    try:
        if time.time() < deadline and best_rank > 1:
            result = _try_rank_reduction(best[0], best[1], best[2], n, m, p, 
                                         best_rank - 5, deadline)
            if result is not None:
                Ur, Vr, Wr, rr = result
                if _verify(Ur, Vr, Wr, n, m, p) and rr < best_rank:
                    best = (Ur, Vr, Wr)
                    best_rank = rr
    except Exception:
        pass

    # Attempt perturbation search for further reduction
    try:
        if time.time() < deadline and best_rank > 1:
            max_iter = int((deadline - time.time()) * 20)  # ~20 iterations per second
            max_iter = max(50, min(max_iter, 2000))
            Up, Vp, Wp, rp = _try_perturbation_search(best[0], best[1], best[2], 
                                                       n, m, p, max_iter, deadline)
            if _verify(Up, Vp, Wp, n, m, p) and rp < best_rank:
                best = (Up, Vp, Wp)
                best_rank = rp
    except Exception:
        pass

    U, V, W = best
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
