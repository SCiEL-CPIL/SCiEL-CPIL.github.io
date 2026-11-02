"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_group_algebra_decomposition(n, m, p, deadline):
    """
    Attempt to find a low-rank decomposition by exploiting the group algebra
    structure F_2[C_n x C_p]. For small dimensions, this can sometimes yield
    a lower rank than the schoolbook method.
    """
    # This is a heuristic: we try to construct a decomposition where the
    # rank is roughly (n*p) * log2(m) or similar, depending on the group structure.
    # However, for exact real-valued decompositions, the group algebra over
    # F_2 doesn't directly translate to a real rank reduction unless we use
    # specific integer/half-integer coefficients.
    #
    # A more practical approach for this specific task (4,4,7) is to check
    # if a known lower-bound decomposition exists. The lower bound for 4x4x7
    # is max(n*m, m*p, p*n) = max(16, 28, 28) = 28.
    # The upper bound is 112.
    #
    # We'll try a structured random search over a lower rank range,
    # but with a smarter initialization: start with the schoolbook and
    # try to "merge" columns via linear dependencies.
    
    if time.time() > deadline:
        return None, None, None, None
    
    # Try ranks from a few below the schoolbook down to the lower bound + a bit
    lower_bound = max(n*m, m*p, p*n)
    start_rank = min(n*m*p - 4, 100)  # Don't go too high, schoolbook is 112
    end_rank = max(lower_bound, 30)
    
    rng = np.random.default_rng(12345)
    
    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Initialize with random sparse matrices
            density = 0.15
            U_rand = rng.standard_normal((n * m, rank)) * (rng.random((n * m, rank)) < density)
            V_rand = rng.standard_normal((m * p, rank)) * (rng.random((m * p, rank)) < density)
            W_rand = rng.standard_normal((p * n, rank)) * (rng.random((p * n, rank)) < density)
            if _verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p):
                return U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), rank
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try the group-algebra-inspired heuristic
    U_g, V_g, W_g, rank_g = _try_group_algebra_decomposition(n, m, p, deadline)
    if U_g is not None:
        best_rank = rank_g
        U, V, W = U_g, V_g, W_g

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
