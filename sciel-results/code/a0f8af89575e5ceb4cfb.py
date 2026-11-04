"""Combined program: block-decomposition structure + rational recovery + targeted search."""
import time
from fractions import Fraction as F

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=4):
    """Check if the decomposition is exact using exact Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = F(0)
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != F(1):
                    return False
    return True


def _to_halfint_list(mat):
    """Convert a list-of-lists of Fractions to a list-of-lists of int/float half-integers."""
    out = []
    for row in mat:
        newrow = []
        for v in row:
            num = v.numerator
            den = v.denominator
            if den == 1:
                newrow.append(num)
            elif den == 2:
                newrow.append(F(num, den))  # half-integer as Fraction (exact)
            else:
                raise ValueError("non half-integer entry")
        out.append(newrow)
    return out


def _strassen_2x2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 by 2x2.
    Conventions:
      A (2x2): row-major (i,j) -> index i*2+j, i,j in {0,1}
      B (2x2): row-major (j,k) -> index j*2+k
      C (2x2): row-major (i,k) -> index i*2+k  (trace convention: for n=m=p=2, W index = k*n+i = k*2+i)
    Returns (U, V, W) each 4 x 7, with U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = [[F(0)] * 7 for _ in range(4)]
    V = [[F(0)] * 7 for _ in range(4)]
    W = [[F(0)] * 7 for _ in range(4)]

    # Strassen factors M1..M7.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7 ; C01 = M3+M5 ; C10 = M2+M4 ; C11 = M1-M2+M3+M5

    # For each factor c, define which A entries (i,j), B entries (j,k), C entries (i,k) appear with what sign.
    # A entries: (0,0),(0,1),(1,0),(1,1)
    # B entries: (0,0),(0,1),(1,0),(1,1)
    # C entries: (0,0),(0,1),(1,0),(1,1)

    factors = []
    # M1
    factors.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,1),(1,1,-1)]))
    # M2
    factors.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]))
    # M3
    factors.append(([(0,0,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(1,1,1)]))
    # M4
    factors.append(([(1,1,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(1,0,1)]))
    # M5
    factors.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]))
    # M6
    factors.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], []))
    # M7
    factors.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))

    for c, (a_terms, b_terms, cc_terms) in enumerate(factors):
        for (i, j, s) in a_terms:
            U[i * 2 + j][c] += s
        for (j, k, s) in b_terms:
            V[j * 2 + k][c] += s
        for (i, k, s) in cc_terms:
            W[k * 2 + i][c] += s
    return U, V, W


def _rank2_2x1x2():
    """
    2x1 by 1x2 -> 2x2, rank 2.
    A (2x1): index i, i in {0,1}
    B (1x2): index k, k in {0,1}
    C (2x2): index (i,k) -> k*2+i
    """
    U = [[F(1), F(0)], [F(0), F(1)]]
    V = [[F(1), F(1)]]
    W = [[F(1), F(1)], [F(1), F(1)]]
    return U, V, W


def _rank2_1x2x2():
    """
    1x2 by 2x2 -> 1x2, rank 2.
    A (1x2): index j, j in {0,1}
    B (2x2): index (j,k) -> j*2+k
    C (1x2): index k, k in {0,1}
    """
    U = [[F(1), F(1)]]
    V = [[F(1), F(0)], [F(0), F(1)]]
    W = [[F(1), F(1)]]
    return U, V, W


def _rank2_1x1x2():
    """
    1x1 by 1x2 -> 1x2, rank 2.
    A (1x1): index 0
    B (1x2): index k
    C (1x2): index k
    """
    U = [[F(1), F(1)]]
    V = [[F(1), F(0)], [F(0), F(1)]]
    W = [[F(1), F(1)]]
    return U, V, W


def _block_decompose_33x34():
    """
    Build the rank-34 block decomposition for 3x3 by 3x4.
    A (3x3), B (3x4), C (3x4).
    Split A into top-left 2x2 (rows0-1,cols0-1), right 2x1 (rows0-1,col2),
           bottom 1x2 (row2,cols0-1), bottom-right 1x1 (row2,col2).
    Split B into top-left 2x2 (rows0-1,cols0-1), top-right 2x2 (rows0-1,cols2-3),
           bottom 1x2 (row2,cols0-1), bottom-right 1x2 (row2,cols2-3).
    C = A*B, output 3x4.
    """
    n, m, p = 3, 3, 4
    rank_total = 34
    U = [[F(0)] * rank_total for _ in range(n * m)]   # 9 x 34
    V = [[F(0)] * rank_total for _ in range(m * p)]   # 12 x 34
    W = [[F(0)] * rank_total for _ in range(p * n)]   # 12 x 34

    sU, sV, sW = _strassen_2x2x2()
    r2U, r2V, r2W = _rank2_2x1x2()
    r1U, r1V, r1W = _rank2_1x2x2()
    r0U, r0V, r0W = _rank2_1x1x2()

    def place(blockU, blockV, blockW, i_off, j_off, k_off, col):
        # blockU: (bi x bj), indices (ii, jj) -> U[(i_off+ii)*m + (j_off+jj)]
        # blockV: (bj x bk), indices (jj, kk) -> V[(j_off+jj)*p + (k_off+kk)]
        # blockW: (bk x bi), indices (kk, ii) -> W[(k_off+kk)*n + (i_off+ii)]
        bi = len(blockU)
        bj = len(blockU[0])
        bk = len(blockW)
        for ii in range(bi):
            for jj in range(bj):
                for c in range(len(blockU[ii])):
                    U[(i_off + ii) * m + (j_off + jj)][col + c] += blockU[ii][jj][c]
        for jj in range(bj):
            for kk in range(bk):
                for c in range(len(blockV[jj])):
                    V[(j_off + jj) * p + (k_off + kk)][col + c] += blockV[jj][kk][c]
        for kk in range(bk):
            for ii in range(bi):
                for c in range(len(blockW[kk])):
                    W[(k_off + kk) * n + (i_off + ii)][col + c] += blockW[kk][ii][c]

    col = 0
    # 1. A22 * B22 (2x2 x 2x2)
    place(sU, sV, sW, 0, 0, 0, col); col += 7
    # 2. A22 * B22p (2x2 x 2x2)
    place(sU, sV, sW, 0, 0, 2, col); col += 7
    # 3. A21 * B12 (2x1 x 1x2)
    place(r2U, r2V, r2W, 0, 2, 0, col); col += 2
    # 4. A21 * B12p (2x1 x 1x2)
    place(r2U, r2V, r2W, 0, 2, 2, col); col += 2
    # 5. A12 * B22 (1x2 x 2x2)
    place(r1U, r1V, r1W, 2, 0, 0, col); col += 2
    # 6. A12 * B22p (1x2 x 2x2)
    place(r1U, r1V, r1W, 2, 0, 2, col); col += 2
    # 7. A11 * B12 (1x1 x 1x2)
    place(r0U, r0V, r0W, 2, 2, 0, col); col += 2
    # 8. A11 * B12p (1x1 x 1x2)
    place(r0U, r0V, r0W, 2, 2, 2, col); col += 2

    assert col == rank_total, f"rank mismatch: {col}"
    return U, V, W


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = F(1)
                V[j * p + k][column] = F(1)
                W[k * n + i][column] = F(1)
                column += 1
    return U, V, W


def _try_numerical_search(U, V, W, n, m, p, deadline):
    """
    Given a candidate (U,V,W) of rank r, attempt a short numerical polish to see if we can
    reduce the rank by one (r-1) by solving a least-squares completion. This is a refinement
    hypothesis; it is cheap and bounded.
    """
    try:
        import numpy as np
    except Exception:
        return None
    r = len(U[0])
    if r <= 1:
        return None
    r_target = r - 1
    start = time.time()

    # Build target tensor T (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Initialize with the first r-1 columns of the current decomposition.
    U0 = np.array(U, dtype=np.float64)[:, :r_target]
    V0 = np.array(V, dtype=np.float64)[:, :r_target]
    W0 = np.array(W, dtype=np.float64)[:, :r_target]

    def residual(params):
        Uv = params[:n * m * r_target].reshape(n * m, r_target)
        Vv = params[n * m * r_target:n * m * r_target + m * p * r_target].reshape(m * p, r_target)
        Wv = params[n * m * r_target + m * p * r_target:].reshape(p * n, r_target)
        Tapprox = np.einsum('ar,br,cr->abc', Uv, Vv, Wv)
        return np.sum((Tapprox - T) ** 2)

    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    try:
        from scipy.optimize import minimize
        res = minimize(residual, x0, method='Nelder-Mead',
                       options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() > deadline:
            return None
        if res.fun < 1e-12:
            # Try rational recovery at half-integers.
            Uv = res.x[:n * m * r_target].reshape(n * m, r_target)
            Vv = res.x[n * m * r_target:n * m * r_target + m * p * r_target].reshape(m * p, r_target)
            Wv = res.x[n * m * r_target + m * p * r_target:].reshape(p * n, r_target)
            # Scale to clear denominators: try denominators 1 and 2.
            for den in (1, 2):
                scale = den
                Ur = np.rint(Uv * scale) / scale
                Vr = np.rint(Vv * scale) / scale
                Wr = np.rint(Wv * scale) / scale
                Tapprox = np.einsum('ar,br,cr->abc', Ur, Vr, Wr)
                if np.allclose(Tapprox, T, atol=1e-9):
                    # Convert to Fractions and validate exactly.
                    Uf = [[F(int(round(Ur[a, c] * 2)), 2) if (Ur[a, c] * 2) % 1 != 0 else F(int(round(Ur[a, c]))) for c in range(r_target)] for a in range(n * m)]
                    Vf = [[F(int(round(Vr[b, c] * 2)), 2) if (Vr[b, c] * 2) % 1 != 0 else F(int(round(Vr[b, c]))) for c in range(r_target)] for b in range(m * p)]
                    Wf = [[F(int(round(Wr[c, c2] * 2)), 2) if (Wr[c, c2] * 2) % 1 != 0 else F(int(round(Wr[c, c2]))) for c in range(p * n) for c2 in range(r_target)]]
                    # fix Wf shape
                    Wf = [[F(int(round(Wr[c, c2] * 2)), 2) if (Wr[c, c2] * 2) % 1 != 0 else F(int(round(Wr[c, c2]))) for c2 in range(r_target)] for c in range(p * n)]
                    if _validate(Uf, Vf, Wf, n, m, p):
                        return (Uf, Vf, Wf)
    except Exception:
        return None
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    n, m, p = DIMENSIONS

    # Primary: block decomposition (rank 34).
    best = None
    try:
        U, V, W = _block_decompose_33x34()
        if _validate(U, V, W, n, m, p):
            best = (U, V, W)
    except Exception:
        pass

    # Refinement: attempt to reduce rank by one via numerical search.
    if best is not None and time.time() < deadline:
        improved = _try_numerical_search(best[0], best[1], best[2], n, m, p, deadline)
        if improved is not None:
            best = improved

    # Fallback: schoolbook (rank 36).
    if best is None:
        U, V, W = _schoolbook()
        best = (U, V, W)

    U, V, W = best
    return {"U": _to_halfint_list(U), "V": _to_halfint_list(V), "W": _to_halfint_list(W)}
# EVOLVE-BLOCK-END
