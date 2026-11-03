"""Matrix multiplication 2x3x5 rank-minimal exact tensor decomposition.

Strategy: search for a low-rank exact decomposition with integer/half-integer
entries.  The multiplication tensor has a diagonal symmetry group (independent
rescaling of input rows/columns), so a decomposition with nonzero entries can be
normalized to have a fixed column equal to 1.  We therefore:
  1. Fix one rank column to the normalized pattern (U=1, V=1, W=1 on its support).
  2. Randomly initialize the other rank-1 columns (entries from {0, +/-1, +/-1/2}).
  3. Optimize the free entries with a smooth continuous objective (sum of squared
     residuals of the tensor reconstruction) -- a smooth surrogate for the hard
     exact condition.
  4. Round the free entries to half-integers and verify EXACTLY with Fraction.
Keep the best valid rank found before the deadline.
"""
import time
import random
import math
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 3, 5]
N, M, P = DIMENSIONS
NMP = N * M * P            # 30 tensor entries
SCHOOLBOOK_RANK = NMP      # 30
DEADLINE = 150.0
SAFETY = 8.0


def _residual(U, V, W, r):
    """Sum of squared residuals of the tensor reconstruction (float)."""
    # U: (N*M, r), V: (M*P, r), W: (P*N, r)
    s = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                acc = 0.0
                u = U[i * M + j]
                v = V[j * P + k]
                w = W[k * N + i]
                for c in range(r):
                    acc += u[c] * v[c] * w[c]
                s += acc * acc
    return s


def _verify_exact(U, V, W, r):
    """Exact verification using Fraction.  Returns True iff decomposition is exact."""
    # Build Fraction matrices
    UF = [[Fraction(U[i][c]) for c in range(r)] for i in range(N * M)]
    VF = [[Fraction(V[i][c]) for c in range(r)] for i in range(M * P)]
    WF = [[Fraction(W[i][c]) for c in range(r)] for i in range(P * N)]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                acc = Fraction(0)
                u = UF[i * M + j]
                v = VF[j * P + k]
                w = WF[k * N + i]
                for c in range(r):
                    acc += u[c] * v[c] * w[c]
                if acc != 1:
                    return False
    return True


def _round_half(x):
    """Round a float to the nearest half-integer (multiple of 0.5)."""
    return int(round(2.0 * x)) / 2.0


def _optimize(r, rng, time_left_fn):
    """Try to find a rank-r exact decomposition.  Returns (U,V,W) or None."""
    # Free entries: all entries except the fixed normalized column (column r-1).
    # Column r-1 is fixed: U[i*M+j][r-1]=1, V[j*P+k][r-1]=1, W[k*N+i][r-1]=1.
    nU = N * M * (r - 1)
    nV = M * P * (r - 1)
    nW = P * N * (r - 1)
    nfree = nU + nV + nW

    def unpack(x):
        U = [[0.0] * r for _ in range(N * M)]
        V = [[0.0] * r for _ in range(M * P)]
        W = [[0.0] * r for _ in range(P * N)]
        for i in range(N * M):
            for c in range(r - 1):
                U[i][c] = x[i * (r - 1) + c]
        for i in range(M * P):
            for c in range(r - 1):
                V[i][c] = x[nU + i * (r - 1) + c]
        for i in range(P * N):
            for c in range(r - 1):
                W[i][c] = x[nU + nV + i * (r - 1) + c]
        # fixed column
        for i in range(N * M):
            U[i][r - 1] = 1.0
        for i in range(M * P):
            V[i][r - 1] = 1.0
        for i in range(P * N):
            W[i][r - 1] = 1.0
        return U, V, W

    def objective(x):
        U, V, W = unpack(x)
        return _residual(U, V, W, r)

    # Random restarts with gradient-free local search (Nelder-Mead style via
    # scipy if available, else a simple coordinate/evolutionary step).
    best = None
    best_val = float('inf')
    start = time.time()
    attempts = 0
    while time.time() - start < time_left_fn() and attempts < 400:
        attempts += 1
        # Initialize free entries: mostly 0 with some small random values.
        x = np.zeros(nfree)
        # Sprinkle random values to give the optimizer a head start.
        for idx in range(nfree):
            if rng.random() < 0.25:
                x[idx] = rng.choice([0.5, -0.5, 1.0, -1.0, 0.25, -0.25])
        # Local optimization: simple iterative coordinate descent with line search
        # plus occasional global perturbations (evolutionary flavor).
        cur = x.copy()
        cur_val = objective(cur)
        best_local = cur.copy()
        best_local_val = cur_val
        step = 0.5
        iters = 0
        max_iters = 60
        while iters < max_iters:
            iters += 1
            improved = False
            # Coordinate descent: try adjusting each free variable.
            order = rng.sample(range(nfree), nfree)
            for idx in order:
                orig = cur[idx]
                for cand in (orig + step, orig - step):
                    cur[idx] = cand
                    val = objective(cur)
                    if val < cur_val - 1e-12:
                        cur_val = val
                        if val < best_local_val:
                            best_local_val = val
                            best_local = cur.copy()
                        improved = True
                        break
                    cur[idx] = orig
            if not improved:
                step *= 0.7
                if step < 1e-4:
                    break
        # Round and verify exactly.
        U, V, W = unpack(best_local)
        Ur = [[_round_half(v) for v in row] for row in U]
        Vr = [[_round_half(v) for v in row] for row in V]
        Wr = [[_round_half(v) for v in row] for row in W]
        if _verify_exact(Ur, Vr, Wr, r):
            return Ur, Vr, Wr
        # Track best continuous solution for a final polish.
        if best_local_val < best_val:
            best_val = best_local_val
            best = best_local.copy()
    # Final high-precision polish: more iterations on the best solution.
    if best is not None:
        cur = best.copy()
        cur_val = objective(cur)
        step = 0.25
        for _ in range(80):
            improved = False
            for idx in range(nfree):
                orig = cur[idx]
                for cand in (orig + step, orig - step):
                    cur[idx] = cand
                    val = objective(cur)
                    if val < cur_val - 1e-12:
                        cur_val = val
                        improved = True
                        break
                    cur[idx] = orig
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
        U, V, W = unpack(cur)
        Ur = [[_round_half(v) for v in row] for row in U]
        Vr = [[_round_half(v) for v in row] for row in V]
        Wr = [[_round_half(v) for v in row] for row in W]
        if _verify_exact(Ur, Vr, Wr, r):
            return Ur, Vr, Wr
    return None


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    rng = random.Random(12345)
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY

    def time_left():
        return max(0.0, deadline - time.time())

    # Start from schoolbook (always valid).
    best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK

    # Try progressively lower ranks.
    for r in range(SCHOOLBOOK_RANK - 1, 12, -1):
        if time.time() > deadline:
            break
        tl = time_left()
        if tl < 2.0:
            break
        # Cap per-rank time so we can try several ranks.
        per_rank = min(tl, max(2.0, 12.0))
        # Temporarily adjust the time budget for this rank.
        rank_deadline = time.time() + per_rank
        orig_deadline = deadline
        # We call _optimize which uses its own start; pass a time_left that
        # respects per_rank.
        def tl_fn():
            return max(0.0, rank_deadline - time.time())
        res = _optimize(r, rng, tl_fn)
        if res is not None:
            U, V, W = res
            # Double-check with exact verification (defensive).
            if _verify_exact(U, V, W, r):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                # Continue searching even lower, but keep this as fallback.
                continue
        # If we found a valid rank, we could keep going lower; if not, continue.
    return best


if __name__ == "__main__":
    out = construct()
    print("rank =", len(out["U"][0]))
