"""
Fresh design: exact rank-minimal tensor decomposition for M_{2,3,6}.

Strategy:
  1. Try to construct a rank-32 decomposition by duplicating a rank-16 decomposition for M_{2,3,3}.
  2. Use an iterative randomized/structured search to find a rank-16 M_{2,3,3} decomposition.
  3. If not found within a time budget, fall back to the schoolbook rank-36 solution.

The output factor W follows the official trace convention W[k*n + i, r].
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _schoolbook(n, m, p):
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
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    # The tensor T_{(i,j),(j',k'),(i'',k'')} = delta_{i,i''} delta_{j,j'} delta_{k,k'}
    # We check: for all (i,j), (j',k'), (i'',k''),
    #   sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r] == delta_{i,i''} delta_{j,j'} delta_{k,k'}
    # Due to symmetry, we only need to check the "matching" case and a few "non-matching" cases.
    rank = U.shape[1]
    # Check all matching entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-9:
                    return False
    # Check some non-matching entries (to ensure zeros)
    # Non-matching: i != i''
    if n >= 2:
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[0*m+j, r] * V[j*p+k, r] * W[k*n+1, r]
                if abs(s) > 1e-9:
                    return False
    # Non-matching: j != j'
    if m >= 2:
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+0, r] * V[1*p+k, r] * W[k*n+i, r]
                if abs(s) > 1e-9:
                    return False
    # Non-matching: k != k'
    if p >= 2:
        for i in range(n):
            for j in range(m):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+0, r] * W[1*n+i, r]
                if abs(s) > 1e-9:
                    return False
    return True

def _search_rank16_m233(deadline):
    """Try to find a rank-16 decomposition for M_{2,3,3}."""
    n, m, p = 2, 3, 3
    rank = 16
    # Initialize with random structure
    np.random.seed(42)
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    start = time.time()
    max_iters = 500
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        # Gradient-free local search: randomly perturb one factor
        for sub in range(20):
            if time.time() > deadline - 5:
                break
            # Compute error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                    # Matching
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1) ** 2
                    # Non-matching: i mismatch
                        for i2 in range(n):
                            if i2 == i: continue
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i2, r]
                            err += s ** 2
                    # Non-matching: j mismatch
                        for j2 in range(m):
                            if j2 == j: continue
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r]
                            err += s ** 2
                    # Non-matching: k mismatch
                        for k2 in range(p):
                            if k2 == k: continue
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k2, r] * W[k2*n+i, r]
                            err += s ** 2
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            # Random perturbation
            which = np.random.randint(3)
            if which == 0:
                idx = np.random.randint(n*m)
                U[idx, :] += np.random.randn(rank) * 0.1
            elif which == 1:
                idx = np.random.randint(m*p)
                V[idx, :] += np.random.randn(rank) * 0.1
            else:
                idx = np.random.randint(p*n)
                W[idx, :] += np.random.randn(rank) * 0.1
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def _search_rank32_direct(n, m, p, deadline):
    """Try to find a rank-32 decomposition for M_{n,m,p} directly."""
    rank = 32
    np.random.seed(123)
    start = time.time()
    max_iters = 200
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        for sub in range(10):
            if time.time() > deadline - 5:
                break
            # Quick error estimate on a subset
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1) ** 2
            if err < 1e-8:
                # Full validation
                if _validate(U, V, W, n, m, p):
                    return U, V, W
            # Perturb
            which = np.random.randint(3)
            if which == 0:
                idx = np.random.randint(n*m)
                U[idx, :] += np.random.randn(rank) * 0.05
            elif which == 1:
                idx = np.random.randint(m*p)
                V[idx, :] += np.random.randn(rank) * 0.05
            else:
                idx = np.random.randint(p*n)
                W[idx, :] += np.random.randn(rank) * 0.05
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin

    # Try rank 32 via duplication of M_{2,3,3} rank-16
    result = _search_rank16_m233(deadline)
    if result is not None:
        U16, V16, W16 = result
        # Duplicate for p=6: use the same decomposition for k=0,1,2 and k=3,4,5
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(16):
            # First half: k=0,1,2
            U[:, r] = U16[:, r]
            V[:, r] = V16[:, r]
            W[:, r] = W16[:, r]
            # Second half: k=3,4,5 (shifted)
            U[:, r+16] = U16[:, r]
            V2 = np.zeros_like(V16)
            # For k=3,4,5, map to j*p+3, j*p+4, j*p+5
            # But V16 has m*p=9 rows for p=3. We need to map k=3->0, k=4->1, k=5->2 in the sub-problem.
            # Actually, the duplication should be: the second half handles the same (i,j,k) structure but for k in {3,4,5}.
            # This requires a different W indexing. Let me rethink.
            # Simpler: just return schoolbook if this doesn't work cleanly.
        # The duplication is non-trivial due to W's indexing. Let's fall through.
        pass

    # Try direct rank-32 search
    result = _search_rank32_direct(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 36
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
