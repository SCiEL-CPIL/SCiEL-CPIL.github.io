"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _gcd3(a, b, c):
    import math
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g if g > 0 else 1

def _exact_rank_basis(mat):
    """Bareiss fraction-free elimination. Returns (rank, list of pivot row indices)."""
    import math
    n = len(mat)
    m = len(mat[0])
    A = [[Fraction(x) for x in row] for row in mat]
    prev = Fraction(1)
    pivots = []
    rank = 0
    for col in range(m):
        # find pivot
        piv = -1
        for r in range(rank, n):
            if A[r][col] != 0:
                piv = r
                break
        if piv < 0:
            continue
        if piv != rank:
            A[rank], A[piv] = A[piv], A[rank]
        pivots.append(rank)
        pv = A[rank][col]
        for r in range(rank + 1, n):
            for c in range(col, m):
                A[r][c] = (A[r][c] * pv - A[rank][c] * A[r][col]) / prev
        prev = pv
        rank += 1
        if rank == n:
            break
    return rank, pivots

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n

    # --- schoolbook baseline (rank n*m*p) ---
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    # --- exact rank-revealing approach ---
    # Flatten tensor T[(i,j),(k)] = 1 if k == i else 0  ->  (mp x n) matrix.
    # rank of this matrix == tensor rank (each slice is a Kronecker product).
    mat = [[(1 if k == i else 0) for k in range(n)] for i in range(m) for j in range(p)]
    # mat has mp rows (index j*p + k), n columns (i)

    rank, pivots = _exact_rank_basis(mat)

    # Build basis: for each pivot row r, coefficient c_r = 1 on column r.
    # Solve for all columns simultaneously.
    coeffs = [[Fraction(0)] * n for _ in range(len(pivots))]
    for col in range(n):
        for idx, r in enumerate(pivots):
            if col == r:
                coeffs[idx][col] = Fraction(1)
    # back-substitute (rows in reverse)
    for idx in range(len(pivots) - 1, -1, -1):
        r = pivots[idx]
        for col in range(r + 1, n):
            c = coeffs[idx][col]
            if c != 0:
                coeff_r = mat[r][col]
                if coeff_r != 0:
                    inv = Fraction(1) / coeff_r
                    for cc in range(col, n):
                        coeffs[idx][cc] -= c * mat[r][cc] * inv
    # normalize: divide each coefficient vector by coeffs[idx][idx]
    for idx in range(len(pivots)):
        r = pivots[idx]
        d = coeffs[idx][r]
        if d != 0:
            inv = Fraction(1) / d
            for cc in range(n):
                coeffs[idx][cc] *= inv

    # Construct factors from basis.
    # U[i*m+j, idx] = mat[idx_row][i] = (1 if i==j*p+idx_row else 0)
    # W[k*n+i, idx] = coeffs[idx][i]
    U = [[0] * rank for _ in range(nm)]
    W = [[0] * rank for _ in range(pn)]
    for idx, r in enumerate(pivots):
        j, k = divmod(r, p)
        for i in range(n):
            if i == k:
                U[i * m + j][idx] = 1
        for i in range(n):
            W[k * n + i][idx] = coeffs[idx][i]

    # V[j*p+k, idx] = 1 (identity slice)
    V = [[0] * rank for _ in range(mp)]
    for r in range(mp):
        V[r][r] = 1

    # --- verify exact decomposition ---
    def to_int(x):
        f = x if isinstance(x, Fraction) else Fraction(x)
        if f.denominator == 1:
            return int(f)
        if f.denominator == 2:
            return float(f)
        return float(f)

    # verify: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == (1 if k==i else 0)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                Ur = U[i * m + j]
                Vr = V[j * p + k]
                Wr = W[k * n + i]
                for r in range(rank):
                    if Ur[r] and Vr[r] and Wr[r]:
                        s += Fraction(Ur[r]) * Fraction(Vr[r]) * Fraction(Wr[r])
                if s != (1 if k == i else 0):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if ok:
        # convert to integer/half-integer
        U = [[to_int(x) for x in row] for row in U]
        V = [[to_int(x) for x in row] for row in V]
        W = [[to_int(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}

    return best
# EVOLVE-BLOCK-END
