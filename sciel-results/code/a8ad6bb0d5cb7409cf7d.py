"""Matrix multiplication tensor 2x5 x 5x5 - rank-minimal exact decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Exact verification of the matmul tensor decomposition.
    T[a,b,c] = 1 iff (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k, else 0.
    T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t].
    """
    rank = len(U[0])
    Um = np.asarray(U, dtype=float)
    Vm = np.asarray(V, dtype=float)
    Wm = np.asarray(W, dtype=float)
    # Non-zero entries must be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(Um[a], np.multiply(Vm[b], Wm[c])))
                if abs(val - 1.0) > tol:
                    return False
    # Zero entries must be 0. Check all (small: 10*25*10 = 2500).
    nm, mp, pn = n * m, m * p, p * n
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            j2 = b // p
            k = b % p
            for c in range(pn):
                k2 = c // n
                i2 = c % n
                if i == i2 and j == j2 and k == k2:
                    continue
                val = float(np.dot(Um[a], np.multiply(Vm[b], Wm[c])))
                if abs(val) > tol:
                    return False
    return True


def _random_search(n, m, p, max_rank, deadline, rng):
    """Randomized search for an exact rank-r decomposition for r from 1..max_rank.
    Returns best verified (U,V,W) or None."""
    nm, mp, pn = n * m, m * p, p * n
    best = None
    best_rank = None
    for r in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            U = rng.standard_normal((nm, r))
            V = rng.standard_normal((mp, r))
            W = rng.standard_normal((pn, r))
            # Normalize columns so product scale is controlled
            for t in range(r):
                norm = 1.0
                U[:, t] /= norm
            # Quick cheap check of the non-zero entries before full verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = float(np.dot(U[a], np.multiply(V[b], W[c])))
                        if abs(val - 1.0) > 1e-7:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break  # found for this r, try to go lower in next loop
        if best_rank is not None and best_rank == r:
            break
    return best


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Guaranteed valid fallback: schoolbook rank n*m*p
    fallback = _schoolbook(n, m, p)
    fallback_rank = n * m * p

    best = fallback
    best_rank = fallback_rank

    rng = random.Random(12345)
    # Attempt to find a strictly lower exact rank.
    # The trivial rank is 50; search below it.
    found = _random_search(n, m, p, max_rank=fallback_rank - 1, deadline=deadline, rng=rng)
    if found is not None:
        found_rank = len(found["U"][0])
        if found_rank < best_rank:
            best = found
            best_rank = found_rank

    return best
# EVOLVE-BLOCK-END
