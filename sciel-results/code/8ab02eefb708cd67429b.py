"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition search."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 120s kill (150s wall-clock)

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, tol=1e-9):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(i0, j0, k0):
    """7-term rank-1 decomposition of the 2x2x2 multiplication sub-block
    starting at (i0,j0,k0). Returns list of (u_vec, v_vec, w_vec) where each
    vector is a dict keyed by (i,j), (j,k), (k,i) respectively."""
    # Strassen's 7 products for [[a,b],[c,d]] @ [[e,f],[g,h]]
    # m1=(a+b)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-c), m5=(a+d)(e+f), m6=(b-c)(g+h), m7=(b-d)(f+h)
    # C11=m1+m2-m4-m7, C12=m3+m4, C21=m5+m6-m1-m3, C22=m1+m2-m5-m6
    terms = []
    # Each term: product of a factor on A, a factor on B, and a factor on C
    # m1: (a+b)(e+h) -> C11+=1, C22+=1
    terms.append(({"(0,0)":1, "(0,1)":1}, {"(0,0)":1, "(0,1)":1}, {"(0,0)":1, "(1,1)":1}))
    # m2: (c+d)e -> C11+=1, C21+=1
    terms.append(({"(1,0)":1, "(1,1)":1}, {"(0,0)":1}, {"(0,0)":1, "(1,0)":1}))
    # m3: a(f-h) -> C12+=1, C21-=1
    terms.append(({"(0,0)":1}, {"(0,1)":1, "(1,1)":1}, {"(0,1)":1, "(1,0)":1}))
    # m4: d(g-c) -> C11-=1, C12+=1
    terms.append(({"(1,1)":1}, {"(1,0)":1, "(0,0)":1}, {"(1,1)":1, "(0,1)":1}))
    # m5: (a+d)(e+f) -> C21+=1, C22-=1
    terms.append(({"(0,0)":1, "(1,1)":1}, {"(0,0)":1, "(0,1)":1}, {"(1,0)":1, "(1,1)":1}))
    # m6: (b-c)(g+h) -> C21+=1, C22-=1
    terms.append(({"(0,1)":1, "(1,0)":1}, {"(1,0)":1, "(1,1)":1}, {"(1,0)":1, "(1,1)":1}))
    # m7: (b-d)(f+h) -> C11-=1, C22-=1
    terms.append(({"(0,1)":1, "(1,1)":1}, {"(0,1)":1, "(1,1)":1}, {"(1,1)":1, "(0,0)":1}))
    return terms

def _build_from_pattern(pattern):
    """pattern: list of (i0,j0,k0) for 2x2x2 Strassen blocks to apply.
    Remaining (i,j,k) use schoolbook single term.
    Returns (U, V, W) as lists."""
    n, m, p = DIMENSIONS
    # Mark which (i,j,k) are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in pattern:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    # Build column lists
    cols_U = []
    cols_V = []
    cols_W = []
    # Strassen blocks
    for (i0, j0, k0) in pattern:
        for (u_dict, v_dict, w_dict) in _strassen_2x2x2(i0, j0, k0):
            u_vec = [0.0]*(n*m)
            v_vec = [0.0]*(m*p)
            w_vec = [0.0]*(p*n)
            for (di, dj), val in u_dict.items():
                u_vec[(i0+di)*m + (j0+dj)] = val
            for (dj, dk), val in v_dict.items():
                v_vec[(j0+dj)*p + (k0+dk)] = val
            for (dk, di), val in w_dict.items():
                w_vec[(k0+dk)*n + (i0+di)] = val
            cols_U.append(u_vec)
            cols_V.append(v_vec)
            cols_W.append(w_vec)
    # Schoolbook for uncovered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_vec = [0.0]*(n*m)
                    v_vec = [0.0]*(m*p)
                    w_vec = [0.0]*(p*n)
                    u_vec[i*m+j] = 1.0
                    v_vec[j*p+k] = 1.0
                    w_vec[k*n+i] = 1.0
                    cols_U.append(u_vec)
                    cols_V.append(v_vec)
                    cols_W.append(w_vec)
    rank = len(cols_U)
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p*n)]
    return U, V, W

def _greedy_eliminate(U, V, W, max_iter=500):
    """Try to remove columns that are linearly redundant (contribute zero net to all T[i,j,k]).
    A column c is removable if for every (i,j,k), the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    sums to zero across... actually we need the sum over remaining columns to still equal 1.
    Simpler: try removing each column and check if the result still verifies.
    This is expensive; do it a limited number of times."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    removed = []
    # Try to remove columns one at a time, re-verifying
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        progress = False
        for c in range(rank - 1, -1, -1):
            if c in removed:
                continue
            # Temporarily remove column c
            U2 = [row[:c] + row[c+1:] for row in U]
            V2 = [row[:c] + row[c+1:] for row in V]
            W2 = [row[:c] + row[c+1:] for row in W]
            if _verify(U2, V2, W2):
                U, V, W = U2, V2, W2
                removed.append(c)
                rank -= 1
                progress = True
                break
        if not progress:
            break
    return U, V, W

def _all_block_placements():
    """All valid (i0,j0,k0) for a 2x2x2 block within [3]x[3]x[8]."""
    n, m, p = DIMENSIONS
    placements = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                placements.append((i0, j0, k0))
    return placements

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    if not _verify(best["U"], best["V"], best["W"]):
        # Should never happen
        return best

    # Search over single Strassen block placements
    placements = _all_block_placements()
    for (i0, j0, k0) in placements:
        if _time_left() < 10.0:
            break
        U, V, W = _build_from_pattern([(i0, j0, k0)])
        if _verify(U, V, W):
            rank = len(U[0])
            if rank < best_rank:
                # Try greedy elimination
                U2, V2, W2 = _greedy_eliminate(U, V, W, max_iter=200)
                if _verify(U2, V2, W2):
                    rank = len(U2[0])
                    best = {"U": U2, "V": V2, "W": W2}
                    best_rank = rank
                else:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank

    # Search over pairs of non-overlapping Strassen blocks
    if _time_left() > 30.0:
        for idx_a in range(len(placements)):
            if _time_left() < 10.0:
                break
            for idx_b in range(idx_a + 1, len(placements)):
                if _time_left() < 5.0:
                    break
                pa = placements[idx_a]
                pb = placements[idx_b]
                # Check non-overlapping
                if (abs(pa[0] - pb[0]) < 2 and abs(pa[1] - pb[1]) < 2 and abs(pa[2] - pb[2]) < 2):
                    continue
                U, V, W = _build_from_pattern([pa, pb])
                if _verify(U, V, W):
                    rank = len(U[0])
                    if rank < best_rank:
                        U2, V2, W2 = _greedy_eliminate(U, V, W, max_iter=100)
                        if _verify(U2, V2, W2):
                            rank = len(U2[0])
                            best = {"U": U2, "V": V2, "W": W2}
                            best_rank = rank
                        else:
                            best = {"U": U, "V": V, "W": W}
                            best_rank = rank

    # Final greedy elimination on the best
    if _time_left() > 10.0:
        U2, V2, W2 = _greedy_eliminate(best["U"], best["V"], best["W"], max_iter=300)
        if _verify(U2, V2, W2):
            best = {"U": U2, "V": V2, "W": W2}

    return best
# EVOLVE-BLOCK-END
