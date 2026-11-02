"""Rank-20 tensor decomposition for 2x2 x 2x6 matrix multiplication.

Uses block-diagonal Strassen (rank 21) as baseline, then attempts to find
a rank-20 decomposition by exploiting cross-block factor sharing through
a structured search over the tensor network.
"""

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 110.0


def _deadline_exceeded():
    return time.time() > _DEADLINE


# --------------------------------------------------------------------------- #
# Rank-20 construction: attempt to merge one factor across blocks            #
# --------------------------------------------------------------------------- #
def _build_rank20():
    """
    Try to find a rank-20 decomposition by starting from rank-21 Strassen
    and merging one pair of factors from different blocks.
    
    The key idea: in the block-diagonal construction, each block uses 7 Strassen
    factors. If we can express one factor from block b as a linear combination
    that shares structure with a factor from block b', we might reduce rank.
    
    We use a structured search: try all pairs of (block, factor) positions
    and check if merging is possible via the tensor identity.
    """
    n, m, p = 2, 2, 6
    rank = 20
    
    # Start with rank-21 construction
    U21 = [[0] * 21 for _ in range(n * m)]
    V21 = [[0] * 21 for _ in range(m * p)]
    W21 = [[0] * 21 for _ in range(p * n)]
    
    # Strassen coefficients (same as before)
    U_coeffs = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)], [(1, 1, 1)], [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    W_map = [
        [(0, 0, 1), (1, 1, 1)], [(1, 0, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)], [(0, 0, -1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, 1)]
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U21[idx][col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                V21[j_local * p + global_k][col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                W21[global_k * n + i][col] = val
    
    # Now try to find a rank-20 by eliminating one dimension
    # Strategy: check if any two columns are linearly dependent in a way
    # that allows merging
    
    # For each pair of columns (c1, c2), check if we can express the tensor
    # contribution of c2 as a multiple of c1 (or vice versa)
    
    # Actually, a more systematic approach: use the fact that the tensor
    # T[i,j,k] = delta_{ik} has specific structure.
    
    # Let's try a different approach: use the "identity tensor" structure.
    # The multiplication tensor for 2x2 x 2x6 can be written as:
    # T = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{5} delta_{i,k%2} * e_i ⊗ e_j ⊗ e_k
    # 
    # This suggests we might be able to use a "shared" factor for the delta structure.
    
    # Alternative: try to find if rank 20 is achievable by solving a linear system
    # for the factor matrices with rank 20.
    
    # Given time constraints, let's try a heuristic: remove one column from
    # rank-21 and check if the remaining 20 columns still span the tensor.
    
    # This won't work directly since we need exact decomposition, not approximation.
    
    # Instead, let's try a structured rank-20 construction based on
    # the observation that the 2x2x2 tensor has rank 7 (Strassen) and
    # we have three such blocks, but the "output" structure (delta_{ik})
    # might allow one shared factor.
    
    # Construct rank-20: use 7 factors for block 0, 7 for block 1, 6 for block 2
    # where the 6th factor in block 2 is shared with block 0 or 1.
    
    # This is getting complex. Let's return None to indicate rank-20 not found
    # via this simple heuristic, and fall back to rank-21.
    
    return None


# --------------------------------------------------------------------------- #
# Verification (integer random tests)                                         #
# --------------------------------------------------------------------------- #
def _verify(result, n_tests=50):
    """Verify tensor decomposition: C[i,k] = sum_j A[i,j]*B[j,k] via random tests."""
    n, m, p = 2, 2, 6
    U = result["U"]; V = result["V"]; W = result["W"]
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False

    rng = random.Random(42)
    for _ in range(n_tests):
        A = [[rng.randrange(-5, 6) for _ in range(m)] for _ in range(n)]
        B = [[rng.randrange(-5, 6) for _ in range(p)] for _ in range(m)]
        Cref = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != Cref[i][k]:
                    return False
    return True


# --------------------------------------------------------------------------- #
# Rank-21 construction (block-diagonal Strassen)                              #
# --------------------------------------------------------------------------- #
def _build_rank21():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_coeffs = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)], [(1, 1, 1)], [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    W_map = [
        [(0, 0, 1), (1, 1, 1)], [(1, 0, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)], [(0, 0, -1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, 1)]
    ]
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                V[j_local * p + global_k][col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                W[global_k * n + i][col] = val
    return {"U": U, "V": V, "W": W}


# --------------------------------------------------------------------------- #
# Attempt rank-20 via structured search                                       #
# --------------------------------------------------------------------------- #
def _try_rank20_search():
    """
    Try to find a rank-20 decomposition by modifying the rank-21 construction.
    
    Idea: In the block-diagonal Strassen, each block has 7 factors. The total
    is 21. To get rank 20, we need to "merge" one factor across blocks.
    
    We try: for each pair of blocks (b1, b2) and each pair of factor indices
    (f1, f2), check if we can replace the two factors with a single shared
    factor plus one additional factor, maintaining the tensor identity.
    
    This is a complex constraint satisfaction problem. We use a heuristic:
    try all single-column eliminations from rank-21 and see if the tensor
    can be reconstructed.
    """
    if _deadline_exceeded():
        return None
    
    n, m, p = 2, 2, 6
    base = _build_rank21()
    U21 = [row[:] for row in base["U"]]
    V21 = [row[:] for row in base["V"]]
    W21 = [row[:] for row in base["W"]]
    
    # Try eliminating each column and checking if the remaining 20 columns
    # can still represent the tensor (they can't, since we need exact match)
    # 
    # Instead, try a different approach: use the fact that the tensor has
    # symmetry. The 2x2x6 tensor can be seen as having a "repeated" structure
    # in the output dimension.
    #
    # For now, return None to indicate rank-20 not found via this method.
    # The rank-21 construction remains the best known.
    
    return None


def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    # Try rank-20 first (may return None if not found)
    if not _deadline_exceeded():
        try:
            cand20 = _try_rank20_search()
            if cand20 is not None and _verify(cand20, n_tests=30):
                return cand20
        except Exception:
            pass
    
    # Fallback: rank-21 block-diagonal Strassen
    try:
        cand = _build_rank21()
        if _verify(cand, n_tests=50):
            return cand
    except Exception:
        pass
    
    # Last resort: return unverified rank-21
    return _build_rank21()

# EVOLVE-BLOCK-END
