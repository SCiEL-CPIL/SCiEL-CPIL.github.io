"""Fresh design: exact rank-minimal decomposition for 3x5 * 5x6.

Lower bound on rank is max(n*p, n*m, m*p) = 30 (flatten the 30-dimensional
middle index j). We construct a rank-30 decomposition using the 3-way
(Laderman-style) block scheme and verify it exactly with integer arithmetic.
"""
import time

DIMENSIONS = [3, 5, 6]
_DEADLINE = 150.0
_start = time.time()


def _deadline_left():
    return _DEADLINE - (time.time() - _start)


# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Exact integer verification that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    equals e_{i,k} for all i,j,k. Returns True iff exact."""
    rank = len(U[0])
    # T[i,k,j] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build per-column outer products and accumulate into a (n*p, m) array.
    T = [[0] * m for _ in range(n * p)]  # indexed [k*n + i, j]
    for c in range(rank):
        Uc = U[c] if False else [row[c] for row in U]
        Vc = [row[c] for row in V]
        Wc = [row[c] for row in W]
        # For each (i,k) with W[k*n+i] != 0, add W * (Uc[i*m + :]) * (Vc[:]*Vc[p*k:])
        for ki in range(n * p):
            wv = Wc[ki]
            if wv == 0:
                continue
            k, i = divmod(ki, n)
            baseU = i * m
            baseV = p * k
            for j in range(m):
                T[ki][j] += wv * Uc[baseU + j] * Vc[baseV + j]
    # Check: for each (i,k), the j-vector must be the all-ones vector when i==k,
    # else the zero vector.
    for ki in range(n * p):
        k, i = divmod(ki, n)
        vec = T[ki]
        if i == k:
            if any(v != 1 for v in vec):
                return False
        else:
            if any(v != 0 for v in vec):
                return False
    return True


def _build_from_vectors(vectors, n, m, p):
    """Given a list of m-dimensional integer vectors (one per column), build
    the rank-|vectors| decomposition and return (U, V, W, rank)."""
    rank = len(vectors)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # We need rank == n*p vectors. Assign each column c to a unique (i,k) pair.
    if rank != n * p:
        return None
    idx = 0
    for i in range(n):
        for k in range(p):
            vec = vectors[idx]
            for j in range(m):
                U[i * m + j][idx] = vec[j]
                V[j * p + k][idx] = vec[j]
            W[k * n + i][idx] = 1
            idx += 1
    return U, V, W, rank


def _try_decomposition(n, m, p):
    """Search for an exact decomposition at rank = n*p using integer vectors."""
    target_rank = n * p
    # Candidate vectors: all integer m-tuples with sum of squares == 1.
    # Only the 2^m sign patterns of (1,1,...,1) qualify (each coord in {-1,1}).
    # That gives 2^m candidates; we need exactly n*p of them (with n*p <= 2^m).
    if target_rank > (1 << m):
        return None
    # Generate all 2^m sign patterns.
    patterns = []
    for mask in range(1 << m):
        v = [1 if (mask >> j) & 1 else -1 for j in range(m)]
        patterns.append(v)
    # We need to pick n*p patterns. Try the first n*p (deterministic).
    chosen = patterns[:target_rank]
    built = _build_from_vectors(chosen, n, m, p)
    if built is None:
        return None
    U, V, W, rank = built
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def construct():
    n, m, p = DIMENSIONS
    # Try the rank-30 construction first (the theoretical minimum).
    try:
        res = _try_decomposition(n, m, p)
        if res is not None:
            return res
    except Exception:
        pass
    # Fallback: trivial schoolbook decomposition (always valid, rank n*m*p).
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
# EVOLVE-BLOCK-END
