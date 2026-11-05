"""Search for rank-minimal exact tensor decomposition of 2x4 * 4x7 matrix mult."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def _verify_frac(Uf, Vf, Wf):
    n, m, p = DIMENSIONS
    rank = len(Uf[0]) if Uf else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _build_from_factors(Uf, Vf, Wf):
    n, m, p = DIMENSIONS
    U = [[float(Uf[i][c]) for c in range(len(Uf[0]))] for i in range(len(Uf))]
    V = [[float(Vf[i][c]) for c in range(len(Vf[0]))] for i in range(len(Vf))]
    W = [[float(Wf[i][c]) for c in range(len(Wf[0]))] for i in range(len(Wf))]
    return {"U": U, "V": V, "W": W}

def _try_rank(r, deadline, rng, max_attempts=3000):
    """Try to find a rank-r decomposition. Returns dict or None."""
    n, m, p = DIMENSIONS
    nU = n * m  # 8
    nV = m * p  # 28
    nW = p * n  # 14
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        if attempts > max_attempts:
            break
        Uf = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nU)]
        Vf = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nV)]
        Wf = [[None] * r for _ in range(nW)]
        ok = True
        for k in range(p):
            for i in range(n):
                M = []
                for j in range(m):
                    row = []
                    for c in range(r):
                        row.append(Uf[i*m+j][c] * Vf[j*p+k][c])
                    M.append(row)
                A = [M[j] + [1] for j in range(m)]
                nrow = len(A)
                ncol = r
                Af = [[Fraction(A[j][c]) for c in range(ncol+1)] for j in range(nrow)]
                row = 0
                pivots = []
                for col in range(ncol):
                    pr = None
                    for rr in range(row, nrow):
                        if Af[rr][col] != 0:
                            pr = rr
                            break
                    if pr is None:
                        continue
                    Af[row], Af[pr] = Af[pr], Af[row]
                    pivots.append(col)
                    pv = Af[row][col]
                    for cc in range(col, ncol+1):
                        Af[row][cc] = Af[row][cc] / pv
                    for rr in range(nrow):
                        if rr != row and Af[rr][col] != 0:
                            factor = Af[rr][col]
                            for cc in range(col, ncol+1):
                                Af[rr][cc] = Af[rr][cc] - factor * Af[row][cc]
                    row += 1
                    if row == nrow:
                        break
                for rr in range(row, nrow):
                    if any(Af[rr][c] != 0 for c in range(ncol)):
                        ok = False
                        break
                    if Af[rr][ncol] != 0:
                        ok = False
                        break
                if not ok:
                    break
                sol = [Fraction(0) for _ in range(ncol)]
                pivot_map = {}
                for idx, col in enumerate(pivots):
                    pivot_map[col] = idx
                for col in range(ncol):
                    if col in pivot_map:
                        sol[col] = Af[pivot_map[col]][ncol]
                Wf[k*n+i] = sol
        if not ok:
            continue
        if _verify_frac(Uf, Vf, Wf):
            return _build_from_factors(Uf, Vf, Wf)
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(12345)
    
    # Schoolbook fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try ranks from the lower bound up to schoolbook
    lower = max(n*m, m*p, p*n)  # 28
    for r in range(lower, rank_school + 1):
        if time.time() > deadline - 5.0:
            break
        res = _try_rank(r, deadline, rng, max_attempts=2000)
        if res is not None:
            best = res
            best_rank = r
            break
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
