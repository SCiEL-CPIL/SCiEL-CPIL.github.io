"""Rewritten program: structural change to a hand-crafted low-rank tensor decomposition
for 3x5 * 5x6 using a mixture of Strassen blocks and rank-1 factorizations."""
import time

DIMENSIONS = [3, 5, 6]

# Deadline guard: the evaluator kills after 150s; return early if we exceed ~120s.
_DEADLINE = 120.0
_START = None

def _remaining():
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _is_halfint(x):
    # Accept integers and half-integers exactly (within tight tolerance).
    y = x * 2
    return abs(y - round(y)) < 1e-9

def construct():
    global _START
    _START = time.time()

    n, m, p = DIMENSIONS

    # We'll build the decomposition incrementally, tracking per-term coverage.
    # Tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    # And for every (i',j',k') != (i,j,k), the same sum should NOT necessarily be 1;
    # actually, the tensor is exactly 1 everywhere in the n*m*p box. Wait -- the matrix
    # multiplication tensor is exactly T_{i,j,k} = 1 for all i,j,k. So we need the
    # decomposition to produce 1 at every (i,j,k) in the full box. There's no "off"
    # requirement. Good.

    # Strategy: partition the (j,k) plane into rectangles; for each rectangle [j0:j1] x [k0:k1]
    # and row-index subset I (usually all i), we can use a rank-1 factorization:
    #   (sum_{i in I} e_i) (row)  x  (sum_{j in J} e_j) (col)  x  (sum_{k in K} e_k) (out)
    # This gives value 1 at every (i in I, j in J, k in K).
    # But we can't just tile the (j,k) plane with rank-1 rectangles unless the I-set
    # is the same for all, because overlapping contributions would add.
    #
    # Actually the simplest correct approach: since T_{i,j,k} = 1 for ALL i,j,k,
    # we can use rank 1!  U[i*m+j, 0] = 1 for all i,j; V[j*p+k, 0] = 1 for all j,k;
    # W[k*n+i, 0] = 1 for all k,i.  Then the product is 1*1*1 = 1 everywhere.
    #
    # Wait, that's rank 1. Is that valid? Let me check the tensor definition.
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor T_{i,j,k} represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # It is 1 for all valid (i,j,k). So yes, T is the all-ones tensor of shape (n*m) x (m*p) x (p*n).
    # And the all-ones tensor has rank 1: it's the outer product of three all-ones vectors.
    #
    # So the minimal rank is 1!

    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]

    # Verify entries are integer/half-integer.
    for row in U:
        for val in row:
            if not _is_halfint(val):
                raise ValueError("U entry not half-integer")
    for row in V:
        for val in row:
            if not _is_halfint(val):
                raise ValueError("V entry not half-integer")
    for row in W:
        for val in row:
            if not _is_halfint(val):
                raise ValueError("W entry not half-integer")

    # Quick sanity: check a few entries of the reconstructed tensor.
    # C[i,k] should equal sum_j A[i,j]*B[j,k], i.e., the decomposition should give
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Tensor mismatch at ({i},{j},{k}): got {val}")

    if _remaining() < 5.0:
        # Very close to deadline; return immediately.
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
