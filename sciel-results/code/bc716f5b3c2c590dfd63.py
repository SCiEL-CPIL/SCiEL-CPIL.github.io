"""Matrix multiplication 2,3,9 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Return the 3-tensor T[i, j, k] = 1 (the matrix multiplication tensor)."""
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = Fraction(1)
    return T

def _decompose_to_UVW(n, m, p, U, V, W, rank):
    """U: (n*m, rank), V: (m*p, rank), W: (p*n, rank) as lists of Fractions."""
    Uout = [[Fraction(0)] * rank for _ in range(n * m)]
    Vout = [[Fraction(0)] * rank for _ in range(m * p)]
    Wout = [[Fraction(0)] * rank for _ in range(p * n)]
    for c in range(rank):
        for a in range(n * m):
            Uout[a][c] = U[a, c]
        for b in range(m * p):
            Vout[b][c] = V[b, c]
        for d in range(p * n):
            Wout[d][c] = W[d, c]
    return Uout, Vout, Wout

def _verify(n, m, p, U, V, W, rank):
    """Check T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] exactly."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W, rank

def _random_rank1_construction(n, m, p, rank, rng):
    """Random rank-r factorization (rational)."""
    def rand_mat(rows, cols):
        M = np.zeros((rows, cols), dtype=object)
        for a in range(rows):
            for b in range(cols):
                M[a, b] = Fraction(rng.randint(-2, 2), 1)
        return M
    return rand_mat(n * m, rank), rand_mat(m * p, rank), rand_mat(p * n, rank)

def _greedy_reduce(n, m, p, U, V, W, rank, deadline):
    """Try to eliminate rank-1 terms by finding linear dependencies.
    Returns (U,V,W,new_rank) if reduced, else None."""
    # Represent each rank-1 term's contribution vector over (i,j,k) space.
    # term c contributes outer product of u_c (size n*m), v_c (size m*p), w_c (size p*n)
    # flattened appropriately; but the tensor entry (i,j,k) couples them.
    # We can form, for each c, the tensor slice T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    # If a linear combination of some slices equals zero, we can drop one.
    # Build matrix of slice vectors (flattened) and find a null vector with small support.
    dim = n * m * p
    slices = []
    for c in range(rank):
        vec = np.zeros(dim, dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[k * n * m + i * m + j] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        slices.append(vec)
    # Find a nontrivial integer null vector with minimal support via exhaustive small combos is too big.
    # Instead, if rank > dim, there is always a null vector; use Gaussian elimination.
    if rank <= dim:
        return None
    # Gaussian elimination over Fractions to find one null vector.
    M = np.array(slices, dtype=object).T  # dim x rank
    # Row reduce
    rows = M.shape[0]
    cols = M.shape[1]
    mat = [list(row) for row in M]
    pivot_cols = []
    r = 0
    for c in range(cols):
        if time.time() > deadline:
            return None
        # find pivot
        piv = None
        for i in range(r, rows):
            if mat[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        for i in range(rows):
            if i != r and mat[i][c] != 0:
                f = mat[i][c] / pv
                for cc in range(c, cols):
                    mat[i][cc] = mat[i][cc] - f * mat[r][cc]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    if not free_cols:
        return None
    # pick one free column as parameter 1, others 0
    fc = free_cols[0]
    nullvec = [Fraction(0)] * cols
    nullvec[fc] = Fraction(1)
    for pr, pc in enumerate(pivot_cols):
        nullvec[pc] = -mat[pr][fc]
    # drop column fc (set that term to zero by absorbing)
    # Actually we can just remove the fc-th rank-1 term if the null combination has it with coeff 1
    # and the rest cancel; but the null relation means sum_c nullvec[c]*slice_c = 0.
    # To reduce rank we need to express the fc term as combination of others.
    # New rank = rank-1 by eliminating fc: for each other c, add nullvec[c] * term_c to ... 
    # This changes U,V,W nontrivially; simpler: keep all but drop fc and adjust is complex.
    # Given complexity, we only reduce when rank>dim trivially; skip otherwise.
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 100.0  # safety margin before 120s kill / 150s wall
    rng = random.Random(12345)

    best_U, best_V, best_W, best_rank = _schoolbook(n, m, p)

    # Try to find a lower-rank exact decomposition via random search + verification.
    # Lower bound on rank: max(n*m, m*p, n*p) is a heuristic; use dim-based.
    dim = n * m * p
    # Attempt a few ranks below schoolbook using random init and exact check.
    for rank in range(max(n * p, m * p), dim):  # search upward from a lower bound
        if time.time() > deadline:
            break
        attempts = 20
        for _ in range(attempts):
            if time.time() > deadline:
                break
            U, V, W = _random_rank1_construction(n, m, p, rank, rng)
            if _verify(n, m, p, U, V, W, rank):
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
        if best_rank < dim:
            break  # found improvement; could try lower but keep simple

    if time.time() > deadline:
        U, V, W, rank = best_U, best_V, best_W, best_rank
    else:
        U, V, W, rank = best_U, best_V, best_W, best_rank

    # Convert to plain lists of ints/floats (Fractions are acceptable but convert to keep clean)
    def to_list(M):
        return [[float(x) if x.denominator != 1 else int(x) for x in row] for row in M]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
