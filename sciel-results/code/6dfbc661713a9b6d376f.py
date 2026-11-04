"""Fresh design: search for a low-rank exact decomposition of the
2x3 * 3x4 matrix-multiplication tensor using exact rational arithmetic."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return {"U": U, "V": V, "W": W}


def _mat_mul(A, B):
    # A: (rows x cols), B: (cols x rows) -> (rows x rows)
    rowsA = len(A)
    colsB = len(B)
    colsA = len(A[0])
    C = [[Fraction(0)] * colsB for _ in range(rowsA)]
    for r in range(rowsA):
        Ar = A[r]
        Cr = C[r]
        for k in range(colsA):
            a = Ar[k]
            if a == 0:
                continue
            Bk = B[k]
            for s in range(colsB):
                Cr[s] += a * Bk[s]
    return C


def _rank_of(M):
    # Gaussian elimination over Fractions; M is list of rows (each a list).
    if not M:
        return 0
    rows = [list(r) for r in M]
    R = len(rows)
    C = len(rows[0])
    rank = 0
    pivot_col = -1
    for col in range(C):
        # find pivot
        pivot = -1
        for r in range(rank, R):
            if rows[r][col] != 0:
                pivot = r
                break
        if pivot < 0:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        piv_val = rows[rank][col]
        # normalize pivot row
        inv = Fraction(1, 1) / piv_val
        prow = rows[rank]
        for c in range(col, C):
            prow[c] *= inv
        # eliminate
        for r in range(R):
            if r == rank:
                continue
            factor = rows[r][col]
            if factor == 0:
                continue
            rr = rows[r]
            for c in range(col, C):
                rr[c] -= factor * prow[c]
        rank += 1
        if rank == R:
            break
    return rank


def _try_rank(r, rng, deadline, best):
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    # Build list of (i,j,k) -> (Urow, Vrow, Wcol)
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    T = len(triples)

    for _ in range(1000000):
        if time.time() >= deadline:
            break
        # random U, V with small Fraction entries
        U = [[rng.randint(-1, 1) for _ in range(r)] for _ in range(N)]
        V = [[rng.randint(-1, 1) for _ in range(r)] for _ in range(M)]
        # Build coefficient matrix: T rows, r cols; rhs: T rows
        A = []
        rhs = []
        for (ur, vr, wc) in triples:
            row = [Fraction(U[ur][c]) * V[vr][c] for c in range(r)]
            A.append(row)
            rhs.append(wc)
        # Solve A W = rhs (W: r x P). Use Gaussian elimination on augmented (r x (r+P)).
        # Augmented matrix: rows = r, cols = r + P
        aug = []
        for c in range(r):
            row = [Fraction(1) if cc == c else Fraction(0) for cc in range(r)]
            for t in range(T):
                row.append(A[t][c])
            aug.append(row)
        # Now solve: for each output column wc, we need W[:,wc] such that A Wcol = rhs_wc.
        # Actually the system is: for each t, sum_c A[t][c] * W[c][wc] = rhs[t].
        # So W is r x P where column wc is solution of A x = rhs.
        # We need A invertible. Solve A X = R where R is r x P with R[c][wc] = rhs_wc if c==wc.
        # Equivalently, find X = A^{-1} R. Compute A^{-1} via elimination, then multiply.
        # Build augmented [A | I]
        aug = []
        for t in range(r):
            row = [Fraction(A[t][c]) for c in range(r)] + [Fraction(1) if cc == t else Fraction(0) for cc in range(r)]
            aug.append(row)
        # Eliminate
        pivots = []
        for col in range(r):
            pivot = -1
            for rr in range(col, r):
                if aug[rr][col] != 0:
                    pivot = rr
                    break
            if pivot < 0:
                return None  # singular
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            inv = Fraction(1, 1) / pv
            prow = aug[col]
            for cc in range(col, 2 * r):
                prow[cc] *= inv
            for rr in range(r):
                if rr == col:
                    continue
                f = aug[rr][col]
                if f == 0:
                    continue
                rrrow = aug[rr]
                for cc in range(col, 2 * r):
                    rrrow[cc] -= f * prow[cc]
            pivots.append(col)
        # aug left half is identity now; right half is A^{-1}
        invA = [aug[t][r + c] for t in range(r) for c in range(r)]
        # W[c][wc] = sum_t invA[c][t] * rhs[t]  (since A W[:,wc] = rhs, W[:,wc] = A^{-1} rhs)
        W = []
        for c in range(r):
            row = []
            for wc in range(P):
                s = Fraction(0)
                for t in range(r):
                    s += invA[c * r + t] * Fraction(rhs[t])
                row.append(s)
            W.append(row)
        # Verify exactly
        ok = True
        for (ur, vr, wc) in triples:
            s = Fraction(0)
            for c in range(r):
                s += Fraction(U[ur][c]) * Fraction(V[vr][c]) * W[c][wc]
            if s != Fraction(wc):
                ok = False
                break
        if ok:
            # Convert to ints/half-ints representation (keep as Fraction, evaluator accepts)
            return {"U": U, "V": V, "W": W}
    return None


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s / 150s kill
    best = _schoolbook()
    best_rank = sum(1 for _ in best["U"][0])
    n, m, p = DIMENSIONS
    # Try ranks from a plausible lower bound up to schoolbook-1
    # Lower bound: max(n*m, m*p, p*n) = 12
    lb = max(n * m, m * p, p * n)
    rng = random.Random(12345)
    for r in range(lb, best_rank):
        if time.time() >= deadline:
            break
        res = _try_rank(r, rng, deadline, best)
        if res is not None:
            best = res
            best_rank = r
            # found a valid lower rank; could try to go lower but we iterate upward
            # continue searching for even lower? We go from lb upward, so first found is minimal in this scan
            break
    return best
# EVOLVE-BLOCK-END
