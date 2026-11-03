import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]


def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    def _time_left():
        return DEADLINE - time.time()

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 12, 12

    # --- Build target tensor and its support (nonzero entries) ---
    T = np.zeros((nm, mp, pn))
    support = []  # list of (a, b, c) where T == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                support.append((a, b, c))
    support = np.array(support, dtype=int)  # (36, 3)

    # Zero-entry constraint list: all (a,b,c) with j1 != j2 OR k1 != i2
    zero_entries = []
    for a in range(nm):
        i1 = a // m
        j1 = a % m
        for b in range(mp):
            j2 = b // p
            k2 = b % p
            if j1 == j2:
                continue
            for c in range(pn):
                zero_entries.append((a, b, c))
    for a in range(nm):
        i1 = a // m
        j1 = a % m
        for b in range(mp):
            j2 = b // p
            k2 = b % p
            if j1 != j2:
                continue
            for c in range(pn):
                k1 = c // n
                i2 = c % n
                if k1 != i2:
                    zero_entries.append((a, b, c))
    zero_entries = np.array(zero_entries, dtype=int)

    def _schoolbook():
        rank = n * m * p  # 36
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify_exact(U, V, W):
        """U,V,W are lists of lists of Fraction. Check half-integer + exact tensor."""
        for mat in (U, V, W):
            for row in mat:
                for val in row:
                    if (val * 2).denominator != 1:
                        return False
        rank = len(U[0])
        # nonzero entries must equal 1
        for (a, b, c) in support:
            s = Fraction(0, 1)
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 1:
                return False
        # zero entries must equal 0
        for (a, b, c) in zero_entries:
            s = Fraction(0, 1)
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 0:
                return False
        return True

    def _round_half_int(mat):
        """Round numpy matrix entries to nearest half-integer, return list of lists of Fraction."""
        rows, cols = mat.shape
        out = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x = mat[i, j]
                # nearest half-integer
                h = round(x * 2.0) / 2.0
                row.append(Fraction(h).limit_denominator(1000000))
            out.append(row)
        return out

    def _residual_on_support(U, V, W):
        """Return max abs error over support (nonzero) and zero entries."""
        R = U.shape[1]
        max_err = 0.0
        for (a, b, c) in support:
            s = 0.0
            for r in range(R):
                s += U[a, r] * V[b, r] * W[c, r]
            e = abs(s - 1.0)
            if e > max_err:
                max_err = e
        for (a, b, c) in zero_entries:
            s = 0.0
            for r in range(R):
                s += U[a, r] * V[b, r] * W[c, r]
            e = abs(s)
            if e > max_err:
                max_err = e
        return max_err

    def _als(U, V, W, R, max_iters, tol):
        """Alternating least squares. U(nm,R), V(mp,R), W(pn,R)."""
        # Precompute per-row target slices
        for it in range(max_iters):
            if _time_left() < 5.0:
                break
            # Update U rows
            for a in range(nm):
                # A of shape (mp*pn, R), A[b*pn+c, r] = V[b,r]*W[c,r]
                A = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, R)
                y = T[a].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
            # Update V rows
            for b in range(mp):
                A = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, R)
                y = T[:, b, :].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
            # Update W rows
            for c in range(pn):
                A = np.einsum('ar,br->abr', U, V).reshape(nm * mp, R)
                y = T[:, :, c].reshape(-1)
                sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            # residual check (cheap on support)
            if _residual_on_support(U, V, W) < tol:
                break
        return U, V, W

    def _gradient_polish(U, V, W, max_iters=200, tol=1e-13):
        """Gradient descent on the full tensor for exactness."""
        R = U.shape[1]
        lr = 0.005
        for it in range(max_iters):
            if _time_left() < 3.0:
                break
            Tcalc = np.zeros((nm, mp, pn))
            for r in range(R):
                Tcalc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            diff = Tcalc - T
            gU = np.einsum('abc,br,cr->ar', diff, V, W)
            gV = np.einsum('abc,ar,cr->br', diff, U, W)
            gW = np.einsum('abc,ar,br->cr', diff, U, V)
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            if np.linalg.norm(diff) < tol:
                break
            if it % 25 == 0:
                lr = max(1e-4, lr * 0.85)
        return U, V, W

    def _try_exact_from_floats(U, V, W):
        """Round to half-integers and verify exactly. Return dict or None."""
        Uf = _round_half_int(U)
        Vf = _round_half_int(V)
        Wf = _round_half_int(W)
        if _verify_exact(Uf, Vf, Wf):
            return {
                "U": [[float(x) for x in row] for row in Uf],
                "V": [[float(x) for x in row] for row in Vf],
                "W": [[float(x) for x in row] for row in Wf],
            }
        return None

    def _init_random(R, scale=1.0):
        U = np.random.randn(nm, R) * scale
        V = np.random.randn(mp, R) * scale
        W = np.random.randn(pn, R) * scale
        return U, V, W

    def _init_structured(R):
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        for r in range(R):
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        U += 0.2 * np.random.randn(nm, R)
        V += 0.2 * np.random.randn(mp, R)
        W += 0.2 * np.random.randn(pn, R)
        return U, V, W

    # --- Search over ranks, ascending from the lower bound ---
    best = None
    for R in range(12, 37):
        if _time_left() < 12.0:
            break
        inits = ['random', 'structured', 'random', 'structured', 'random', 'random']
        for itype in inits:
            if _time_left() < 6.0:
                break
            if itype == 'random':
                U, V, W = _init_random(R)
            else:
                U, V, W = _init_structured(R)
            U, V, W = _als(U, V, W, R, max_iters=40, tol=1e-9)
            if _residual_on_support(U, V, W) < 1e-7:
                U, V, W = _gradient_polish(U, V, W, max_iters=150, tol=1e-13)
                res = _try_exact_from_floats(U, V, W)
                if res is not None:
                    return res
        # If we're at a promising rank and still have time, do extra random restarts
        if R <= 20 and _time_left() > 20.0:
            for _ in range(4):
                if _time_left() < 6.0:
                    break
                U, V, W = _init_random(R, scale=0.7)
                U, V, W = _als(U, V, W, R, max_iters=60, tol=1e-9)
                if _residual_on_support(U, V, W) < 1e-8:
                    U, V, W = _gradient_polish(U, V, W, max_iters=200, tol=1e-14)
                    res = _try_exact_from_floats(U, V, W)
                    if res is not None:
                        return res

    # Fallback: guaranteed-exact schoolbook decomposition
    return _schoolbook()
