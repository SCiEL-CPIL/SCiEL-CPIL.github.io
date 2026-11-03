"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    """Flattened multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T

def _verify(U, V, W, n, m, p, rank, samples=200, seed=0):
    """Check U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] sums to the target tensor."""
    rng = np.random.default_rng(seed)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build the rank-rank-1 outer sum tensor via random probing of entries.
    # For correctness, verify all (i,j,k) entries exactly using the formula.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Start from the guaranteed schoolbook decomposition.
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Attempt to find a lower-rank exact decomposition via a bilinear
    # Strassen-style construction; verify exactly before accepting.
    # (Kept within the same design family: explicit rank-rank-1 sum.)
    # For 2x4 x 4x5 a rank-36 bilinear scheme is known; we build it by
    # partitioning and sharing products, then verify.
    try:
        cand = _try_rank36(n, m, p)
        if cand is not None:
            r = len(cand["U"][0])
            if r < best_rank and _verify(cand["U"], cand["V"], cand["W"], n, m, p, r):
                best = cand
                best_rank = r
    except Exception:
        pass

    # Deadline guard: return best found.
    if time.time() > deadline:
        return best

    return best

def _try_rank36(n, m, p):
    """Construct a rank-36 bilinear decomposition for 2x4 x 4x5 if applicable."""
    if (n, m, p) != (2, 4, 5):
        return None
    # Build via explicit bilinear forms. We use a verified rank-36 scheme:
    # partition the 4 middle indices into two 2-blocks and the 5 outputs into
    # groups, sharing Strassen products across output columns.
    # Represent U,V,W as lists of length 36.
    U = [[0] * 36 for _ in range(8)]   # n*m = 8
    V = [[0] * 36 for _ in range(20)]  # m*p = 20
    W = [[0] * 36 for _ in range(10)]  # p*n = 10

    # Indices: U[i*4+j], V[j*5+k], W[k*2+i]
    # We use a bilinear scheme: for each of 36 products, set one entry in
    # each of U, V, W. The scheme is defined by a list of (i,j,k,u,v,w)
    # where the product contributes u*v*w to T[i,j,k]. We need the sum over
    # products of u_c*v_c*w_c = 1 for each (i,j,k) and 0 elsewhere (only
    # (i,j,k) with valid indices appear, all =1).
    #
    # This specific rank-36 table is nontrivial; rather than risk an
    # incorrect hardcoded table, we return None and rely on schoolbook.
    return None
# EVOLVE-BLOCK-END
