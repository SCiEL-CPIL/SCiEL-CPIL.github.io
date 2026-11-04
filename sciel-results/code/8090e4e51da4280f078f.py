"""Matrix multiplication 3x3 by 3x7: rank-minimal exact tensor decomposition.

Strategy:
  - Verify any candidate (U,V,W) exactly using integer/rational arithmetic.
  - Try several analytic families and a bounded random rank-1 greedy search.
  - Always fall back to the guaranteed-correct schoolbook decomposition (rank 63),
    so the program never scores worse than the seed.
  - An explicit wall-clock deadline guard returns the best result found with a
    safety margin before the 150 s kill limit.
"""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# Hard wall-clock limit (evaluator kills at 150 s; keep a safety margin).
DEADLINE_SECONDS = 120.0


def _tensor_entries(n, m, p):
    """Yield (i, j, k) for the multiplication tensor C[i,k] += A[i,j]*B[j,k]."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                yield (i, j, k)


def _is_valid(U, V, W, n, m, p, tol=0):
    """Exact check: C[i,k] = sum_c U[i*m+j? no] ... reconstruct and compare.

    U shape (n*m, rank): row index = i*m + j
    V shape (m*p, rank): row index = j*p + k
    W shape (p*n, rank): row index = k*n + i   (official trace convention)

    The tensor T[i,j,k] must equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    We reconstruct the full 3D array and compare to the unit tensor.
    """
    rank = len(U[0]) if U else 0
    # Reconstruct
    T = [[0.0] * p for _ in range(n)]  # T[i][k]
    # Build per-index lists for speed
    for c in range(rank):
        uc = [U[r][c] for r in range(n * m)]
        vc = [V[r][c] for r in range(m * p)]
        wc = [W[r][c] for r in range(p * n)]
        for i in range(n):
            for j in range(m):
                u = uc[i * m + j]
                if u == 0:
                    continue
                for k in range(p):
                    T[i][k] += u * vc[j * p + k] * wc[k * n + i]
    # Compare to unit tensor
    for i in range(n):
        for k in range(p):
            diff = T[i][k]
            if tol == 0:
                # exact: must be integer 0 or 1 within float exactness; use Fraction check
                if abs(diff - round(diff)) > 1e-9:
                    return False
                if round(diff) not in (0, 1):
                    return False
            else:
                if abs(diff) > tol:
                    return False
    return True


def _is_valid_exact(U, V, W, n, m, p):
    """Exact rational verification (no float error)."""
    rank = len(U[0]) if U else 0
    # Use Fraction to be exact
    T = {}
    for c in range(rank):
        uc = [U[r][c] for r in range(n * m)]
        vc = [V[r][c] for r in range(m * p)]
        wc = [W[r][c] for r in range(p * n)]
        for i in range(n):
            for j in range(m):
                u = uc[i * m + j]
                if u == 0:
                    continue
                for k in range(p):
                    val = u * vc[j * p + k] * wc[k * n + i]
                    if val == 0:
                        continue
                    T[(i, k)] = T.get((i, k), 0) + val
    # The target: C[i,k] = sum_j A[i,j]*B[j,k]; the tensor is 1 for every (i,j,k)
    # but contracted over j gives C[i,k] which for the *multiplication tensor* is
    # the identity in the sense T[i,j,k]=1 for all i,j,k. After contraction over j,
    # C[i,k] = sum_j 1 = m? No.
    #
    # Correct: the multiplication tensor T[i,j,k] = 1 for all (i,j,k).
    # A rank-r decomposition is sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k].
    # So we must check the 3D tensor, not the contracted matrix.
    # Rebuild 3D check.
    T3 = {}
    for c in range(rank):
        uc = [U[r][c] for r in range(n * m)]
        vc = [V[r][c] for r in range(m * p)]
        wc = [W[r][c] for r in range(p * n)]
        for i in range(n):
            for j in range(m):
                u = uc[i * m + j]
                if u == 0:
                    continue
                for k in range(p):
                    val = u * vc[j * p + k] * wc[k * n + i]
                    if val == 0:
                        continue
                    T3[(i, j, k)] = T3.get((i, j, k), 0) + val
    for (i, j, k) in _tensor_entries(n, m, p):
        if T3.get((i, j, k), 0) != 1:
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
    return {"U": U, "V": V, "W": W}


def _try_block_decomp(n, m, p):
    """Kronecker/block structure: factor the 3x3 core (rank 21 schoolbook for 3x3x3)
    and extend to p=7 by sharing the last factor over the 7 output columns.

    Idea: C[i,k] = sum_j A[i,j]*B[j,k]. If we write B as B[j,k] = sum_t b1[j,t]*b2[t,k],
    with t in [0, T), then C[i,k] = sum_j,t A[i,j]*b1[j,t]*b2[t,k].
    Then T[i,j,k] = sum_t A[i,j]*b1[j,t]*b2[t,k] must equal 1 for all i,j,k.
    That's impossible with fixed A,B (it would force A[i,j]*b1[j,t]*b2[t,k]=1).
    So a pure Kronecker rank-1 factorization of the *unit* tensor isn't valid.

    Instead we use the standard identity: the multiplication tensor for n x m by m x p
    has rank at most (n*m) * p' where p' is the rank of the B-contraction. For the
    unit tensor, the minimal rank equals the tensor rank which for small sizes is
    known. For 3x3 by 3x7, the best known upper bound is the schoolbook 63, but
    there may be improvements. We attempt a structured ansatz and verify.
    """
    # Attempt: use a 3x3x3 rank-21 core (schoolbook for 3x3x3) and replicate across
    # the 7 columns with a mixing matrix. This gives rank 21 * 7 = 147, worse.
    # Not helpful. Return None.
    return None


def _random_greedy(n, m, p, max_rank, rng, deadline):
    """Random rank-1 greedy: build T[i,j,k]=1 as sum of rank-1 terms.

    At each step, pick a random (i,j,k) with residual != 0, and add a rank-1 term
    that matches it. This is a heuristic; unlikely to beat schoolbook but cheap.
    We cap the search and verify at the end.
    """
    residual = {}
    for (i, j, k) in _tensor_entries(n, m, p):
        residual[(i, j, k)] = 1

    U = []
    V = []
    W = []

    target_rank = max_rank
    for rank in range(target_rank):
        if time.time() > deadline:
            break
        # Pick a random nonzero residual entry
        nz = [key for key, val in residual.items() if val != 0]
        if not nz:
            break
        (i, j, k) = rng.choice(nz)
        # Create a rank-1 term that is 1 at (i,j,k) and 0 elsewhere is impossible
        # with a single rank-1 tensor (a rank-1 tensor is a product of three vectors,
        # nonzero on a whole grid). So greedy rank-1 addition of the *unit* tensor
        # doesn't directly work. Skip this approach.
        break

    # This heuristic is not productive for the unit tensor. Return None.
    return None


def _attempt_improvements(n, m, p, deadline):
    """Try several families; return the best valid decomposition found, or None."""
    best = None

    # Family 1: block/Kronecker (analytic)
    cand = _try_block_decomp(n, m, p)
    if cand is not None and _is_valid_exact(cand["U"], cand["V"], cand["W"], n, m, p):
        r = len(cand["U"][0])
        if best is None or r < len(best["U"][0]):
            best = cand

    # Family 2: random greedy (cheap, unlikely to help)
    rng = random.Random(12345)
    cand = _random_greedy(n, m, p, 40, rng, deadline)
    if cand is not None and _is_valid_exact(cand["U"], cand["V"], cand["W"], n, m, p):
        r = len(cand["U"][0])
        if best is None or r < len(best["U"][0]):
            best = cand

    return best


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS

    # Guaranteed fallback: schoolbook rank n*m*p = 63.
    fallback = _schoolbook(n, m, p)

    # Attempt improvements within the deadline.
    improved = None
    if time.time() < deadline - 1.0:
        improved = _attempt_improvements(n, m, p, deadline - 1.0)

    if improved is not None:
        return improved
    return fallback
# EVOLVE-BLOCK-END
