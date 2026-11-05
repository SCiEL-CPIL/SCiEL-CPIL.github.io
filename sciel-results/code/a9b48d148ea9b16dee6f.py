"""Search for a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy: start from the schoolbook identity (rank 105), then iteratively look for
"rank-1 subtraction" moves: a small local perturbation of a few factor columns that
reduces the total rank while preserving the tensor identity. Each candidate is
verified exactly against the tensor before being accepted.
"""
import numpy as np
import itertools
import time

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
NP, NM, NPp = n * p, m * n, p * m

DEADLINE = 115.0  # safety margin below 120 s
_deadline = time.time() + DEADLINE


def _build_tensor():
    """Return the matrix-multiplication tensor M[i, j, k] = 1."""
    M = np.ones((n, m, p), dtype=object)
    return M


def _tensor_from_factors(U, V, W, rank):
    """Compute the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]."""
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        ucol = np.array(U[:, c], dtype=object)
        vcol = np.array(V[:, c], dtype=object)
        wcol = np.array(W[:, c], dtype=object)
        for i in range(n):
            for j in range(m):
                uval = ucol[i * m + j]
                if uval == 0:
                    continue
                for k in range(p):
                    vval = vcol[j * p + k]
                    wval = wcol[k * n + i]
                    T[i, j, k] = T[i, j, k] + uval * vval * wval
    return T


def _schoolbook():
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
    return U, V, W, rank


def _validate(U, V, W, rank, target):
    """Return True if factors exactly reproduce the target tensor."""
    T = _tensor_from_factors(U, V, W, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != target[i, j, k]:
                    return False
    return True


def construct():
    global _deadline
    target = _build_tensor()
    U, V, W, rank = _schoolbook()
    best = (rank, U, V, W)

    def try_replace(indices, U_new_rows, V_new_rows, W_new_rows):
        """Try replacing the set of columns `indices` with the given rows (one column each).
        Returns (new_rank, new_U, new_V, new_W) if valid, else None."""
        nonlocal rank
        idx = sorted(indices)
        new_rank = rank - len(idx)  # we remove len(idx) columns, add len(idx) columns -> same? 
        # Actually replacement keeps same rank; a "reduction" removes columns without adding.
        # For a reduction: we drop `len(idx)` columns and add `add_count` columns.
        return None

    # Simple rank reduction: drop a set of columns and add fewer new rank-1 terms
    # such that the tensor is preserved. Search over small subsets.
    while time.time() < _deadline and rank > 60:
        improved = False
        # Try dropping pairs of columns and see if a single new term can compensate
        for a in range(rank):
            if time.time() >= _deadline:
                break
            for b in range(a + 1, rank):
                # Remove columns a and b; the "deficit" tensor D is the sum of those two terms
                # We need to find one rank-1 term that equals D. That's generally impossible,
                # but for certain structured pairs it works (e.g., both terms share a zero pattern).
                # Build deficit tensor.
                D = np.zeros((n, m, p), dtype=object)
                for c in (a, b):
                    ucol = np.array(U[:, c], dtype=object)
                    vcol = np.array(V[:, c], dtype=object)
                    wcol = np.array(W[:, c], dtype=object)
                    for i in range(n):
                        for j in range(m):
                            uval = ucol[i * m + j]
                            if uval == 0:
                                continue
                            for k in range(p):
                                vval = vcol[j * p + k]
                                wval = wcol[k * n + i]
                                D[i, j, k] = D[i, j, k] + uval * vval * wval
                # Check if D is a rank-1 tensor: D[i,j,k] = alpha[i,j]*beta[j,k]*gamma[k,i] form
                # i.e., there exist vectors u', v', w' such that D[i,j,k]=u'[i,j]*v'[j,k]*w'[k,i]
                # We attempt a simple factorization: if D has support on a single (i,j,k) pattern
                # that's a product of row/col/... Actually check: D is rank-1 in the 3-way sense.
                # Cheap necessary test: D must be nonzero at positions forming a "box" product.
                nonzero = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                           if D[i, j, k] != 0]
                if len(nonzero) == 0:
                    continue
                # A 3-way rank-1 tensor has support = product of supports on each mode.
                # Collect supports.
                si = set(i for (i, j, k) in nonzero)
                sj = set(j for (i, j, k) in nonzero)
                sk = set(k for (i, j, k) in nonzero)
                if len(nonzero) != len(si) * len(sj) * len(sk):
                    continue
                # Try to factor: pick a reference index.
                i0, j0, k0 = nonzero[0]
                u0 = D[i0, j0, k0]
                if u0 == 0:
                    continue
                # Assume u'[i0,j0]=1, v'[j0,k0]=1, w'[k0,i0]=u0. Solve for others.
                # u'[i,j0]*w'[k0,i] = D[i,j0,k0] for all i
                # This is a system; check consistency.
                u_row = {}  # (i,j0)
                w_col = {}  # (k0,i)
                # Set w'[k0,i0]=u0, u'[i0,j0]=1, v'[j0,k0]=1.
                # For each i: D[i, j0, k0] = u'[i,j0]*w'[k0,i]
                # We have one equation per i, two unknowns -> free. Choose u'[i,j0]=D[i,j0,k0]/u0, w'[k0,i]=u0 for i != i0; for i=i0: u'[i0,j0]=1,w'[k0,i0]=u0.
                # Then check all positions.
                ok = True
                uarr = {}
                warr = {}
                warr[(k0, i0)] = u0
                uarr[(i0, j0)] = 1
                for i in si:
                    if i == i0:
                        continue
                    uarr[(i, j0)] = D[i, j0, k0] / u0
                    warr[(k0, i)] = u0
                for j in sj:
                    if j == j0:
                        continue
                    for k in sk:
                        pass
                # v'[j,k] = D[i0,j,k] / (u'[i0,j]*w'[k,i0]) but u'[i0,j] unknown for j!=j0.
                # Use D[i0, j0, k] = v'[j0,k]*u0 -> v'[j0,k]=D[i0,j0,k]/u0
                varr = {}
                varr[(j0, k0)] = 1
                for k in sk:
                    if k == k0:
                        continue
                    varr[(j0, k)] = D[i0, j0, k] / u0
                # Now full check: for all (i,j,k) in si x sj x sk:
                # D[i,j,k] = u'[i,j]*v'[j,k]*w'[k,i]
                # We only have partial u' (j=j0) and v' (j=j0) and w' (k=k0).
                # For positions not on the reference lines we cannot determine factors uniquely.
                # So this simple method only works when si={i0} or sj={j0} or sk={k0} (a line).
                if not (len(si) == 1 or len(sj) == 1 or len(sk) == 1):
                    continue
                # If one mode is singleton, we can factor fully.
                try:
                    # Build u, v, w arrays.
                    ufull = np.zeros((n, m), dtype=object)
                    vfull = np.zeros((m, p), dtype=object)
                    wfull = np.zeros((p, n), dtype=object)
                    if len(si) == 1:
                        i = i0
                        for j in sj:
                            for k in sk:
                                if (j, k) == (j0, k0):
                                    continue
                                if D[i, j, k] != 0:
                                    if (i, j) in uarr and uarr[(i, j)] != 0:
                                        vfull[j, k] = D[i, j, k] / (uarr[(i, j)] * wfull[k, i]) if wfull[k, i] != 0 else None
                                        if vfull[j, k] is None:
                                            raise ValueError
                                    else:
                                        ufull[i, j] = 1
                                        vfull[j, k] = D[i, j, k] / wfull[k, i] if wfull[k, i] != 0 else None
                                        if vfull[j, k] is None:
                                            raise ValueError
                        # fill u
                        for j in sj:
                            ufull[i0, j] = uarr.get((i0, j), 1)
                        for k in sk:
                            wfull[k, i0] = warr.get((k, i0), u0)
                        for k in sk:
                            vfull[j0, k] = varr.get((j0, k), 1)
                    # Verify the new term reproduces D exactly.
                    for i in si:
                        for j in sj:
                            for k in sk:
                                val = (ufull[i, j] or 0) * (vfull[j, k] or 0) * (wfull[k, i] or 0)
                                if val != D[i, j, k]:
                                    raise ValueError
                    # Also ensure it is zero outside the box (it will be, since ufull/vfull/wfull
                    # are zero outside si/sj/sk by construction, except where we set values).
                    # Construct candidate: remove columns a,b; add one new column.
                    new_rank = rank - 1
                    nU = [row[:] for row in U]
                    nV = [row[:] for row in V]
                    nW = [row[:] for row in W]
                    # Remove columns a and b.
                    keep = [c for c in range(rank) if c != a and c != b]
                    nU = [[nU[r][c] for c in keep] for r in range(n * m)]
                    nV = [[nV[r][c] for c in keep] for r in range(m * p)]
                    nW = [[nW[r][c] for c in keep] for r in range(p * n)]
                    # Append new column.
                    for r in range(n * m):
                        nU[r].append(ufull[r // m, r % m])
                    for r in range(m * p):
                        nV[r].append(vfull[r // p, r % p])
                    for r in range(p * n):
                        nW[r].append(wfull[r // n, r % n])
                    if _validate(nU, nV, nW, new_rank, target):
                        U, V, W, rank = nU, nV, nW, new_rank
                        if rank < best[0]:
                            best = (rank, [r[:] for r in U], [r[:] for r in V], [r[:] for r in W])
                        improved = True
                        break
                except Exception:
                    pass
            if improved:
                break
        if not improved:
            # Try single-column drops that are "redundant" (a column whose removal still
            # leaves the tensor valid because two columns cancel).
            # This is rare; skip to avoid heavy cost.
            break

    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
