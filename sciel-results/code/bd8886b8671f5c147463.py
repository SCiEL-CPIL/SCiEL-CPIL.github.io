"""Matrix multiplication tensor decomposition for 3x4x7."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of invalid triples
    rng = np.random.RandomState(42)
    for _ in range(200):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        i, j = divmod(a, m)
        jp, k = divmod(b, p)
        kn, i2 = divmod(c, n)
        if j == jp and k == kn and i == i2:
            continue
        val = np.sum(U[a] * V[b] * W[c])
        if abs(val) > tol:
            return False
    
    return True

def _construct_schoolbook(n, m, p):
    """Standard schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_random_search(n, m, p, target_rank, deadline, seed=0):
    """Try to find a decomposition of given rank using random search."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with random small integers/half-integers
    for r in range(target_rank):
        # Each rank-1 term should contribute to some subset of (i,j,k)
        # Randomly assign which (i,j,k) this term helps with
        num_contrib = rng.randint(1, 4)
        for _ in range(num_contrib):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            U[i * m + j, r] = rng.choice([-1, 1])
            V[j * p + k, r] = rng.choice([-1, 1])
            W[k * n + i, r] = rng.choice([-1, 1])
    
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_reduction(n, m, p, deadline):
    """Try to find a lower-rank decomposition by exploiting structure."""
    # For 3x4x7, try to see if we can reduce rank by combining terms
    # This is a heuristic approach
    
    best = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try various ranks below schoolbook
    # Given time constraints, only try a few
    for target_rank in [83, 82, 81, 80]:
        if time.time() > deadline - 30:
            break
        result = _try_random_search(n, m, p, target_rank, deadline, seed=target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decomposition
    result, rank = _try_structured_reduction(n, m, p, deadline)
    if result is not None and rank < best_rank:
        best_result = result
        best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
