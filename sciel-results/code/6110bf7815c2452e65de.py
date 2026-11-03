"""Matrix multiplication 4x4 by 4x6 rank-minimal exact tensor decomposition.

Strategy:
- Start from the guaranteed rank-96 schoolbook.
- Run a deadline-guarded randomized + greedy search over integer/half-integer
  factor entries to find a lower rank that exactly reconstructs the tensor.
- Every candidate rank r is verified by the fast rank-r identity check
  (sum_c U[:,c] * V[:,c] == W as a 3-tensor) before being accepted.
- If nothing better is found, return the best valid decomposition (schoolbook).
"""
import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 6]

# Hard deadline guard: evaluator kills at 150s; keep a safety margin.
DEADLINE_SECONDS = 135.0
_START_TIME = time.time()

def _remaining_seconds():
    return max(0.0, DEADLINE_SECONDS - (time.time() - _START_TIME))

def _build_schoolbook(n=4, m=4, p=6):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _tensor_from_factors(U, V, W):
    """Return the 3-tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    n, m, p = 4, 4, 6
    T = np.zeros((n, m, p), dtype=np.int64)
    # Efficient: loop over columns (rank) and do outer products.
    for c in range(U.shape[1]):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        if not (np.any(uc) and np.any(vc) and np.any(wc)):
            continue
        # uc: (n*m), vc: (m*p), wc: (p*n)
        # For each i,j,k: uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
        # Do it via einsum-like manual loops for clarity/speed at small sizes.
        for i in range(n):
            for j in range(m):
                a = uc[i * m + j]
                if a == 0:
                    continue
                for k in range(p):
                    b = vc[j * p + k]
                    if b == 0:
                        continue
                    d = wc[k * n + i]
                    if d == 0:
                        continue
                    T[i, j, k] += a * b * d
    return T

def _identity_tensor(n=4, m=4, p=6):
    """The matrix-multiplication tensor: T[i,j,k] = 1 iff (i,j,k) corresponds to a
    valid term in (A*B)[i,k] = sum_j A[i,j]*B[j,k]. For 4x4 * 4x6, every (i,j,k)
    with i in [0,4), j in [0,4), k in [0,6) is a valid term, so T[i,j,k]=1 for all."""
    return np.ones((n, m, p), dtype=np.int64)

def _verify_decomposition(U, V, W, T_id):
    """Check if sum_c U[:,c] V[:,c] W[:,c] == T_id exactly."""
    T = _tensor_from_factors(U, V, W)
    return np.array_equal(T, T_id)

def _random_half_int(rng, lo=-2, hi=2):
    """Return a random integer or half-integer in [lo, hi]."""
    val = rng.randint(2 * lo, 2 * hi)
    if val % 2 == 0:
        return val // 2
    else:
        return val / 2.0

def _greedy_rank_peel(U, V, W, T_id, max_rank, rng, time_budget):
    """
    Try to reduce rank by greedily finding rank-1 terms that can replace
    combinations of existing columns. This is heuristic; verify at the end.
    """
    current_rank = U.shape[1]
    if current_rank <= max_rank:
        return U, V, W

    # We'll attempt to find a set of r new columns that reconstruct T_id,
    # starting from r = max_rank and going up.
    # For efficiency, we use a randomized search over sparse half-integer factors.
    best_U, best_V, best_W = U, V, W
    best_r = current_rank

    # Try a few target ranks
    for target_r in range(max_rank, min(current_rank, max_rank + 12) + 1):
        if _remaining_seconds() < 5.0:
            break
        if target_r >= best_r:
            continue

        # Randomized attempts to find a rank-target_r decomposition
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and _remaining_seconds() > 5.0:
            attempts += 1
            # Generate random sparse half-integer factors
            U_r = np.zeros((16, target_r), dtype=np.float64)
            V_r = np.zeros((24, target_r), dtype=np.float64)
            W_r = np.zeros((24, target_r), dtype=np.float64)

            sparsity = 0.3
            for c in range(target_r):
                # Each column is sparse
                for i in range(16):
                    if rng.random() < sparsity:
                        U_r[i, c] = _random_half_int(rng)
                for j in range(24):
                    if rng.random() < sparsity:
                        V_r[j, c] = _random_half_int(rng)
                for k in range(24):
                    if rng.random() < sparsity:
                        W_r[k, c] = _random_half_int(rng)

            # Quick check: does this reconstruct T_id?
            if _verify_decomposition(U_r.astype(np.int64) * 2, V_r.astype(np.int64) * 2, W_r.astype(np.int64) * 2, T_id * 4):
                # Found a valid decomposition at target_r
                best_r = target_r
                best_U = (U_r * 2).astype(np.int64)
                best_V = (V_r * 2).astype(np.int64)
                best_W = (W_r * 2).astype(np.int64)
                break

    return best_U, best_V, best_W

def _structured_search(U, V, W, T_id, rng):
    """
    Deterministic + randomized search over structured patterns that exploit
    the asymmetry (m=4, p=6). Try to find rank-90, 88, 86, 84 decompositions.
    """
    best_U, best_V, best_W = U, V, W
    best_r = U.shape[1]

    # Pattern 1: Group columns by (i,j) pairs and try to share k-dimension
    # For each (i,j), we have 6 k-values. If we can represent 6 unit vectors
    # in a lower-dimensional space, we save rank.
    # Try: for each (i,j), use 4 columns instead of 6 (if possible)
    # This is a heuristic; we'll verify.

    # Randomized structured search
    for target_r in [90, 88, 86, 84, 82, 80]:
        if _remaining_seconds() < 10.0:
            break
        if target_r >= best_r:
            continue

        # Try to find a decomposition at target_r using structured patterns
        # Pattern: use the fact that T[i,j,k]=1 for all i,j,k
        # A rank-r decomposition exists iff we can write the all-ones tensor
        # as a sum of r rank-1 tensors with half-integer entries.

        # Heuristic: try random sparse factors with higher density
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts and _remaining_seconds() > 10.0:
            attempts += 1
            U_r = np.zeros((16, target_r), dtype=np.float64)
            V_r = np.zeros((24, target_r), dtype=np.float64)
            W_r = np.zeros((24, target_r), dtype=np.float64)

            # Higher sparsity for structured patterns
            for c in range(target_r):
                for i in range(16):
                    if rng.random() < 0.5:
                        U_r[i, c] = _random_half_int(rng, -1, 1)
                for j in range(24):
                    if rng.random() < 0.5:
                        V_r[j, c] = _random_half_int(rng, -1, 1)
                for k in range(24):
                    if rng.random() < 0.5:
                        W_r[k, c] = _random_half_int(rng, -1, 1)

            # Verify (scale to integers)
            U_int = (U_r * 2).astype(np.int64)
            V_int = (V_r * 2).astype(np.int64)
            W_int = (W_r * 2).astype(np.int64)
            if _verify_decomposition(U_int, V_int, W_int, T_id * 4):
                best_r = target_r
                best_U = U_int
                best_V = V_int
                best_W = W_int
                break

    return best_U, best_V, best_W

def construct():
    """Return a valid rank-minimal (or best-found) exact tensor decomposition."""
    n, m, p = DIMENSIONS
    T_id = _identity_tensor(n, m, p)

    # Start with schoolbook rank-96
    U, V, W = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]

    # Verify schoolbook
    if not _verify_decomposition(U, V, W, T_id):
        # Should not happen, but handle gracefully
        raise RuntimeError("Schoolbook decomposition failed verification")

    # Seed RNG for reproducibility within a run
    rng = random.Random(42)

    # Try structured search first (more likely to find patterns)
    if _remaining_seconds() > 30.0:
        U2, V2, W2 = _structured_search(U, V, W, T_id, rng)
        if W2.shape[1] < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = W2.shape[1]

    # Try greedy/randomized search if time remains
    if _remaining_seconds() > 20.0:
        U3, V3, W3 = _greedy_rank_peel(U, V, W, T_id, best_rank - 1, rng, _remaining_seconds())
        if W3.shape[1] < best_rank:
            best_U, best_V, best_W = U3, V3, W3
            best_rank = W3.shape[1]

    # Convert to lists for the expected interface
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]

    # Final verification
    U_arr = np.array(U_list, dtype=np.int64)
    V_arr = np.array(V_list, dtype=np.int64)
    W_arr = np.array(W_list, dtype=np.int64)
    if not _verify_decomposition(U_arr, V_arr, W_arr, T_id):
        # Fall back to schoolbook
        U_list = [[int(x) for x in row] for row in U]
        V_list = [[int(x) for x in row] for row in V]
        W_list = [[int(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
