import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]


def _validate_float_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation using numpy."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = W.shape[1]
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True


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
    return U, V, W


def _strassen_block(i0, j0, k0):
    """Return 7 (u,v,w) dict-terms for a 2x2x2 Strassen block."""
    terms = []
    # M1
    terms.append(({(i0, j0): 1, (i0+1, j0+1): 1},
                  {(j0, k0): 1, (j0+1, k0+1): 1},
                  {(k0, i0): 1, (k0+1, i0+1): 1}))
    # M2
    terms.append(({(i0+1, j0): 1, (i0+1, j0+1): 1},
                  {(j0, k0): 1},
                  {(k0, i0+1): 1, (k0+1, i0+1): -1}))
    # M3
    terms.append(({(i0, j0): 1},
                  {(j0, k0+1): 1, (j0+1, k0+1): -1},
                  {(k0+1, i0): 1, (k0+1, i0+1): 1}))
    # M4
    terms.append(({(i0+1, j0+1): 1},
                  {(j0+1, k0): 1, (j0, k0): -1},
                  {(k0, i0): 1, (k0, i0+1): 1}))
    # M5
    terms.append(({(i0, j0): 1, (i0, j0+1): 1},
                  {(j0+1, k0+1): 1},
                  {(k0, i0): -1, (k0+1, i0): 1}))
    # M6
    terms.append(({(i0+1, j0): 1, (i0, j0): -1},
                  {(j0, k0): 1, (j0, k0+1): 1},
                  {(k0+1, i0+1): 1}))
    # M7
    terms.append(({(i0, j0+1): 1, (i0+1, j0+1): -1},
                  {(j0+1, k0): 1, (j0+1, k0+1): 1},
                  {(k0, i0): 1}))
    return terms


def _build_strassen(n, m, p):
    """Build a valid decomposition via Strassen 2x2x2 blocks + schoolbook remainder."""
    terms = []
    # i in {0,1}, j in {0..4}, k in {0..4}
    # j=0,1: k=0,1 Strassen; k=2,3 Strassen; k=4 schoolbook
    terms += _strassen_block(0, 0, 0)
    terms += _strassen_block(0, 0, 2)
    for i in (0, 1):
        for j in (0, 1):
            terms.append(({(i, j): 1}, {(j, 4): 1}, {(4, i): 1}))
    # j=2,3: k=0,1 Strassen; k=2,3 Strassen; k=4 schoolbook
    terms += _strassen_block(0, 2, 0)
    terms += _strassen_block(0, 2, 2)
    for i in (0, 1):
        for j in (2, 3):
            terms.append(({(i, j): 1}, {(j, 4): 1}, {(4, i): 1}))
    # j=4: schoolbook all k
    for i in (0, 1):
        for k in (0, 1, 2, 3, 4):
            terms.append(({(i, 4): 1}, {(4, k): 1}, {(k, i): 1}))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        u, v, w = terms[c]
        for (i, j), val in u.items():
            U[i * m + j][c] = val
        for (j, k), val in v.items():
            V[j * p + k][c] = val
        for (k, i), val in w.items():
            W[k * n + i][c] = val
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, deadline, max_rounds=200):
    """
    Exact rank reduction.

    For each candidate column c, find coefficients a_d (d != c) such that
    adding sum_d a_d * col_d to col_c cancels col_c's contribution at every
    tensor entry (i,j,k) where col_c has nonzero product. If solvable, we can
    replace col_c by (col_c + sum a_d col_d) which has zero contribution
    everywhere, then drop it.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    r = len(W[0])
    zero = Fraction(0, 1)

    while r > 1 and time.time() < deadline:
        reduced = False
        for c in range(r):
            if time.time() > deadline - 3:
                return U, V, W
            # Find all (i,j,k) where col c contributes
            # contribution at (i,j,k) from col c = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # We need: for each such (i,j,k), sum_d a_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = -U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Unknowns: a_d for d in {0..r-1}\{c}
            # Build linear system
            rows = []  # list of (coeff_dict, rhs)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        cu = U[i*m+j][c]
                        cv = V[j*p+k][c]
                        cw = W[k*n+i][c]
                        if cu == 0 or cv == 0 or cw == 0:
                            continue
                        rhs = -(Fraction(cu) * Fraction(cv) * Fraction(cw))
                        coeff = {}
                        for d in range(r):
                            if d == c:
                                continue
                            term = Fraction(U[i*m+j][d]) * Fraction(V[j*p+k][d]) * Fraction(W[k*n+i][d])
                            if term != 0:
                                coeff[d] = coeff.get(d, 0) + term
                        rows.append((coeff, rhs))

            if not rows:
                continue

            # Collect unknowns
            unknowns = set()
            for coeff, rhs in rows:
                unknowns.update(coeff.keys())
            unknowns = sorted(unknowns)
            if not unknowns:
                continue
            idx = {u: t for t, u in enumerate(unknowns)}
            num_eq = len(rows)
            num_unk = len(unknowns)
            if num_eq < num_unk:
                # Underdetermined; still may have solution (a_d=0 works only if rhs=0).
                # If all rhs=0, trivial solution exists -> col c already redundant? 
                # Actually if all rhs=0, then a_d=0 works, meaning col c contributes 0 everywhere
                # which can't happen in a valid decomposition. Skip.
                continue

            # Build matrix A (num_eq x num_unk) and b
            A = [[zero] * num_unk for _ in range(num_eq)]
            b = [zero] * num_eq
            for e, (coeff, rhs) in enumerate(rows):
                for u, val in coeff.items():
                    A[e][idx[u]] = val
                b[e] = rhs

            # Gaussian elimination over Q
            # Solve A x = b
            n_unk = num_unk
            n_eq = num_eq
            # Augmented
            M = [A[e][:] + [b[e]] for e in range(n_eq)]
            rank_m = 0
            pivot_col = {}
            for col in range(n_unk):
                # find pivot row >= rank_m with nonzero M[row][col]
                piv = -1
                for row in range(rank_m, n_eq):
                    if M[row][col] != 0:
                        piv = row
                        break
                if piv == -1:
                    continue
                M[rank_m], M[piv] = M[piv], M[rank_m]
                # normalize
                pv = M[rank_m][col]
                if pv != 1:
                    for cc in range(col, n_unk + 1):
                        M[rank_m][cc] /= pv
                # eliminate other rows
                for row in range(n_eq):
                    if row == rank_m:
                        continue
                    factor = M[row][col]
                    if factor != 0:
                        for cc in range(col, n_unk + 1):
                            M[row][cc] -= factor * M[rank_m][cc]
                pivot_col[col] = rank_m
                rank_m += 1

            # Check consistency: any row with all-zero coeffs but nonzero rhs?
            consistent = True
            for row in range(n_eq):
                if all(M[row][cc] == 0 for cc in range(n_unk)):
                    if M[row][n_unk] != 0:
                        consistent = False
                        break
            if not consistent:
                continue

            # Free variables: columns not in pivot_col -> set to 0
            sol = {u: zero for u in unknowns}
            for col in range(n_unk):
                if col in pivot_col:
                    sol[unknowns[col]] = M[pivot_col[col]][n_unk]
                else:
                    sol[unknowns[col]] = zero

            # Apply: new col c = col c + sum_d a_d col d  (this makes col c contribute 0 everywhere)
            # Actually we want to ABSORB col c into others so we can drop c.
            # The standard trick: define new col c' = col c + sum a_d col d. Then col c'
            # has zero contribution at every (i,j,k) where col c had nonzero (by construction)
            # and zero elsewhere too (since a_d only affects entries, and we verified all).
            # Wait: we need col c' to contribute 0 at EVERY (i,j,k), not just where col c nonzero.
            # At (i,j,k) where col c is zero, col c' contribution = sum a_d * U_d V_d W_d which
            # is NOT necessarily 0. So this approach is flawed unless we also ensure those are 0.
            #
            # Correct approach: We want to find a_d such that for ALL (i,j,k),
            #   U_c V_c W_c + sum_d a_d U_d V_d W_d = 0
            # i.e., the new column c' = c + sum a_d d contributes 0 everywhere,
            # then we can drop c' (it's a zero column) -> rank reduces by 1.
            # This is a system over ALL (i,j,k) with rhs = -U_c V_c W_c (0 where c is 0).
            # Let's rebuild rows over ALL entries.
            rows2 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        cu = U[i*m+j][c]
                        cv = V[j*p+k][c]
                        cw = W[k*n+i][c]
                        rhs = -(Fraction(cu) * Fraction(cv) * Fraction(cw))
                        coeff = {}
                        for d in range(r):
                            if d == c:
                                continue
                            term = Fraction(U[i*m+j][d]) * Fraction(V[j*p+k][d]) * Fraction(W[k*n+i][d])
                            if term != 0:
                                coeff[d] = coeff.get(d, 0) + term
                        rows2.append((coeff, rhs))

            unknowns2 = set()
            for coeff, rhs in rows2:
                unknowns2.update(coeff.keys())
            unknowns2 = sorted(unknowns2)
            if not unknowns2:
                continue
            idx2 = {u: t for t, u in enumerate(unknowns2)}
            num_eq2 = len(rows2)
            num_unk2 = len(unknowns2)
            if num_eq2 < num_unk2:
                continue
            A2 = [[zero] * num_unk2 for _ in range(num_eq2)]
            b2 = [zero] * num_eq2
            for e, (coeff, rhs) in enumerate(rows2):
                for u, val in coeff.items():
                    A2[e][idx2[u]] = val
                b2[e] = rhs
            M2 = [A2[e][:] + [b2[e]] for e in range(num_eq2)]
            rank_m2 = 0
            pivot_col2 = {}
            for col in range(num_unk2):
                piv = -1
                for row in range(rank_m2, num_eq2):
                    if M2[row][col] != 0:
                        piv = row
                        break
                if piv == -1:
                    continue
                M2[rank_m2], M2[piv] = M2[piv], M2[rank_m2]
                pv = M2[rank_m2][col]
                if pv != 1:
                    for cc in range(col, num_unk2 + 1):
                        M2[rank_m2][cc] /= pv
                for row in range(num_eq2):
                    if row == rank_m2:
                        continue
                    factor = M2[row][col]
                    if factor != 0:
                        for cc in range(col, num_unk2 + 1):
                            M2[row][cc] -= factor * M2[rank_m2][cc]
                pivot_col2[col] = rank_m2
                rank_m2 += 1
            consistent2 = True
            for row in range(num_eq2):
                if all(M2[row][cc] == 0 for cc in range(num_unk2)):
                    if M2[row][num_unk2] != 0:
                        consistent2 = False
                        break
            if not consistent2:
                continue
            sol2 = {u: zero for u in unknowns2}
            for col in range(num_unk2):
                if col in pivot_col2:
                    sol2[unknowns2[col]] = M2[pivot_col2[col]][num_unk2]

            # Verify the solution makes col c' a zero column
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                        for d in unknowns2:
                            a = sol2[d]
                            if a != 0:
                                s += a * Fraction(U[i*m+j][d]) * Fraction(V[j*p+k][d]) * Fraction(W[k*n+i][d])
                        if s != 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Now col c is "absorbable": we can drop column c entirely because
            # its contribution can be exactly cancelled by adjusting... 
            # Actually the cleanest: since col c' = c + sum a_d d is a zero column,
            # the decomposition with col c replaced by col c' is still valid (zero col
            # contributes nothing), and then we can drop the zero col c'.
            # But we must also confirm dropping col c (the original) leaves the tensor
            # unchanged ONLY if col c's contribution is exactly matched by the a_d adjustment
            # added to the OTHER columns. Equivalent: set col c = 0 and add a_d * col_d to... 
            # 
            # Simplest valid operation: replace column c by the zero column (since
            # c' = c + sum a_d d = 0  =>  c = -sum a_d d). Then the decomposition
            # becomes: sum over d of (col_d) * (1 + a_d) ... no.
            #
            # Let me think again. Original tensor T = sum_c col_c (where col_c is the
            # rank-1 tensor U_c⊗V_c⊗W_c). We found a_d such that col_c + sum_d a_d col_d = 0.
            # So col_c = -sum_d a_d col_d. Therefore T = sum_{d!=c} col_d + col_c
            # = sum_{d!=c} col_d - sum_d a_d col_d = sum_{d!=c} (1 - a_d) col_d.
            # So we can DROP column c and MULTIPLY each remaining column d by (1 - a_d)!
            # This is a clean rank reduction.
            for d in range(r):
                if d == c:
                    continue
                scale = 1 - sol2[d]
                for i in range(n * m):
                    U[i][d] = U[i][d] * scale
                for j in range(m * p):
                    V[j][d] = V[j][d] * scale
                for k in range(p * n):
                    W[k][d] = W[k][d] * scale
            # Drop column c
            for i in range(n * m):
                U[i].pop(c)
            for j in range(m * p):
                V[j].pop(c)
            for k in range(p * n):
                W[k].pop(c)
            r -= 1
            reduced = True
            break
        if not reduced:
            break
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Start from Strassen-based valid decomposition
    U, V, W = _build_strassen(n, m, p)

    # Verify validity (float, fast)
    if not _validate_float_fast(U, V, W, n, m, p):
        # Fall back to schoolbook (guaranteed valid)
        U, V, W = _schoolbook(n, m, p)

    # Exact rank reduction
    if time.time() < deadline - 10:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)

    # Final exact validation; if it fails (shouldn't), fall back to a known-valid form
    if time.time() < deadline - 5:
        if not _validate_exact(U, V, W, n, m, p):
            U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
