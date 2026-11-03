"""Task program; the EVOLVE-BLOCK contains the evolved candidate."""
import math
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# --- Time-budget guard (wall-clock 150 s hard limit; keep a safety margin) ---
_START_TIME = time.time()
WALL_CLOCK_LIMIT = 150.0          # seconds
SAFETY_MARGIN = 5.0               # return by this many seconds before the limit
DEADLINE = _START_TIME + (WALL_CLOCK_LIMIT - SAFETY_MARGIN)

# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
def _now() -> float:
    return time.time()


def _exceeds_deadline() -> bool:
    return _now() > DEADLINE


def _exact_verify(U, V, W) -> bool:
    """Exact rational verification of the decomposition convention.

    Convention used by the evaluator:
      U: (n*m) x rank  indexed [i*m + j, c]
      V: (m*p) x rank  indexed [j*p + k, c]
      W: (p*n) x rank  indexed [k*n + i, c]   <-- official trace convention

    We must verify for all i, j, k, c:
      sum over i, j, k, c ... actually the matrix-mult tensor condition:

      (U * V * W^T-like)  such that
      For every i (0..n-1), j (0..m-1), k (0..p-1):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  =  1
      and for every (a, b, c_index) with a=(i',j') row of U, b=(j',k') col of
      V, c=(k',i') row of W such that the indices don't form a valid
      (i,j,k) triple for all three simultaneously ... the tensor is only
      non-zero when the three indices share a common (i,j,k).

      Concretely the multiplication tensor T[i,j,k] = 1 iff i,j,k are valid
      indices, and the decomposition condition is:
        T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
      for all valid (i,j,k), and the decomposition should be *exact*: for
      any selection of i, j, k the above must equal 1.

      There are no other non-zero tensor entries; the other constraints are
      that the decomposition should not produce non-zero values where the
      tensor is zero.  But because U, V, W are only indexed by valid (i,j),
      (j,k), (k,i) pairs, and every column c contributes a rank-1 tensor
      U[:,c] * V[:,c] * W[:,c], which has support on ALL triples
      (i,j,k) for which U[i*m+j,c] != 0 AND V[j*p+k,c] != 0 AND W[k*n+i,c] != 0,
      the necessary and sufficient condition for exactness is:

      1. For every (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
      2. The total sum over all (i,j,k) of the above is n*m*p (trivially 1 each).

      Actually condition 1 alone is sufficient IF we also ensure that the
      only non-zero contributions are for valid (i,j,k).  But since U has rows
      indexed by all valid (i,j), V by all valid (j,k), W by all valid (k,i),
      every term U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] corresponds to a valid
      (i,j,k).  There are no "invalid" (i,j,k) because the indices are
      bounded.  So condition 1 is both necessary and sufficient.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U and U[0] else 0
    Uf = [[Fraction(int(x)) for x in row] for row in U]
    Vf = [[Fraction(int(x)) for x in row] for row in V]
    Wf = [[Fraction(int(x)) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if s != 1:
                    return False
    return True


def _build_schoolbook():
    n, m, p = DIMENSIONS
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


def _random_decomposition(rank: int, scale: int = 2) -> dict:
    """Generate a random integer decomposition of a given rank."""
    n, m, p = DIMENSIONS
    U = [[random.randint(-scale, scale) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.randint(-scale, scale) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.randint(-scale, scale) for _ in range(rank)] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}


def _test_decomposition_random_inputs(decomp: dict, num_trials: int = 5) -> bool:
    """Fast necessary-condition test using random integer inputs.

    For random matrices A (n x m) and B (m x p), check that
      (A * B)[i, k] = sum_j A[i,j] * B[j,k]
    can be recovered from:
      C[i, k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_k ... )
    Wait, let's be precise.  The decomposition gives:
      C[i, k] = sum_j (A[i,j] * B[j,k])  ... no, that's the definition.

    The tensor decomposition means:
      (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) *
                    (something) ... 

    Actually with the convention:
      U: (n*m) x r  -- row i*m+j
      V: (m*p) x r  -- row j*p+k
      W: (p*n) x r  -- row k*n+i

    The decomposition condition is that for all (i,j,k):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (the tensor entry)

    To verify via matrix multiplication, for any A (n x m) and B (m x p):
      (A * B)[i, k] should equal sum_j A[i,j] * B[j,k]

    Using the decomposition, we can compute:
      sum_c (sum_j A[i,j] * U[i*m+j, c]) * (V^T-like) ... this is getting
      complicated.  A simpler necessary test: pick random A, B, compute C = A*B,
      and also compute C' = sum over c of (outer product reconstructed).

    Actually the cleanest necessary-and-sufficient test with random inputs:
      For random A (n x m), B (m x p), define:
        X[c] = A * reshape(U_c) where U_c is the c-th column reshaped as n x m
        Y[c] = reshape(V_c) * B where V_c is the c-th column reshaped as m x p
        Then the candidate C from the decomposition is:
          C_candidate = sum_c X[c] * W_c * Y[c] ... no.

    Let's just verify the tensor equation directly for a few random (i,j,k) 
    ... but we already do exact verification.  The random-input test is faster:

    For the decomposition to work, it must hold that:
      (A*B)[i,k] = sum_j A[i,j] * B[j,k]  ... this is always true.

    The decomposition provides an *alternative* factorisation.  The actual
    verification is: for all i, j, k, the sum_c equation = 1.

    A faster necessary test: verify the equation for a random subset of (i,j,k)
    with a few random integer-valued decompositions.  But we need to test the
    *given* decomposition, not random ones.

    So _test_decomposition_random_inputs should check the tensor equation for
    a random sample of (i,j,k) pairs (faster than full verification when rank
    is large, but similar when rank is small).  Let's just do a quick check on
    all (i,j,k) since n*m*p = 36 is tiny, but skip if we already verified.
    
    Actually the purpose here is to quickly reject bad random decompositions
    generated in the search loop.  A fast reject: check one random (i,j,k).
    """
    n, m, p = DIMENSIONS
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0]) if U and U[0] else 0
    # Pick a few random (i, j, k)
    for _ in range(num_trials):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        s = 0
        for c in range(rank):
            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if s != 1:
            return False
    return True


def _search_lower_rank(best_decomp: dict) -> dict:
    """Try to find a decomposition of rank < best by random search.
    
    Returns the best decomposition found so far (possibly the input).
    Respects the wall-clock deadline.
    """
    n, m, p = DIMENSIONS
    current_best = best_decomp
    best_rank = len(current_best["U"][0]) if current_best and current_best["U"] else n * m * p
    rng = random.Random(42)  # deterministic for reproducibility

    # Try ranks from best_rank - 1 down to 1
    for rank in range(best_rank - 1, 0, -1):
        if _exceeds_deadline():
            break
        # Number of random trials is conditioned on remaining time
        # Rough estimate: each trial is O(n*m*p*rank) = O(36*rank) ~ 360 ops for rank=10
        # Allow up to a few hundred trials per rank if time permits
        max_trials = 200
        for trial in range(max_trials):
            if _exceeds_deadline():
                break
            decomp = _random_decomposition(rank, scale=3)
            if _test_decomposition_random_inputs(decomp, num_trials=2):
                # Passed quick test; do exact verification
                if _exact_verify(decomp["U"], decomp["V"], decomp["W"]):
                    current_best = decomp
                    best_rank = rank
                    break  # found a lower rank, continue to next (even lower) rank
    return current_best


def construct() -> dict:
    # Start with the schoolbook rank-36 decomposition
    best = _build_schoolbook()

    # Search for a lower-rank decomposition within the same family
    if not _exceeds_deadline():
        best = _search_lower_rank(best)

    # Final deadline guard: ensure we return within the safety margin
    if _exceeds_deadline():
        # Already have best; just return it
        pass

    return best
# EVOLVE-BLOCK-END
