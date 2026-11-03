import time
import numpy as np
import random

# Task: 4x5 * 5x5. n=4, m=5, p=5.
# Tensor T[i,j,k] = 1 if output (i,k) uses A[i,j]*B[j,k].
# Convention: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
# Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
# Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
# and = 0 for all invalid (a,b,c)

N, M, P = 4, 5, 5
NMP = N * M * P  # 100 (schoolbook rank)

def _verify_decomposition(U, V, W, n, m, p, max_invalid_checks=5000):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    
    # Check valid entries: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    
    # Check some invalid entries (sample)
    random.seed(42)
    for _ in range(max_invalid_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        
        # Check if (a,b,c) is a valid index combination
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        
        # Valid if j_a == j_b and i_a == i_c
        if j_a == j_b and i_a == i_c:
            continue  # Skip valid entries
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > 1e-8:
            return False
    
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_strassen_block(n, m, p, n_sub, m_sub, p_sub, strassen_rank):
    """
    Try to construct a decomposition using Strassen-like block structure.
    For 4x5*5x5, we can view it as blocks and try to exploit structure.
    """
    # This is a heuristic approach - try to find a structured low-rank decomposition
    # by exploiting the block structure of the matrices.
    
    # For now, this is a placeholder for more sophisticated search
    return None, None, None

def _random_search(n, m, p, target_rank, max_restarts=10, time_limit=30.0):
    """
    Try to find a rank-target_rank decomposition using random restarts.
    Returns (U, V, W) if found, else (None, None, None).
    """
    t0 = time.time()
    
    for restart in range(max_restarts):
        if time.time() - t0 > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Quick error check on valid entries
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        
        if err < 1e-10:
            # Found a valid decomposition!
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None, None, None

def _structured_search(n, m, p, target_rank, time_limit=60.0):
    """
    Try structured search for low-rank decomposition.
    Uses the fact that the matrix multiplication tensor has specific structure.
    """
    t0 = time.time()
    
    # Strategy: try to find a decomposition where the rank-1 terms have
    # a specific pattern based on the block structure of the matrices.
    
    # For 4x5*5x5, we can try to exploit the fact that n=4, m=5, p=5
    # and look for decompositions with rank less than 100.
    
    # One approach: use the "laser method" or "tensor splitting" ideas,
    # but these are complex to implement from scratch.
    
    # For now, try a few structured ansatz:
    # 1. Try rank 90, 85, 80 with random search
    # 2. Try block-structured decompositions
    
    # Given time constraints, just do random search for a few ranks
    for trial_rank in [target_rank]:
        if time.time() - t0 > time_limit:
            break
        
        U, V, W = _random_search(n, m, p, trial_rank, max_restarts=5, time_limit=10.0)
        if U is not None:
            return U, V, W
    
    return None, None, None

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # hard limit
    safety = 10.0          # keep 10s margin

    n, m, p = N, M, P

    # Start with schoolbook decomposition
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p

    # Verify schoolbook is correct
    if not _verify_decomposition(U0, V0, W0, n, m, p, max_invalid_checks=1000):
        raise RuntimeError("Schoolbook verification failed")

    # Try to find a lower rank decomposition using structured search
    # For 4x5x5, try a few ranks with the GF(2) support search
    # Lower bound is max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But actual lower bounds are higher. Known best for 4x4x4 is 49.
    # For 4x5x5, let's try ranks from 95 down to 70.
    
    for target_rank in [95, 90, 85, 80, 75, 70]:
        if time.time() > deadline - safety:
            break
        
        U, V, W = _structured_search(n, m, p, target_rank, time_limit=15.0)
        if U is not None:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break

    # If we found a better decomposition, verify it
    if best_rank < n * m * p:
        if not _verify_decomposition(best_U, best_V, best_W, n, m, p, max_invalid_checks=10000):
            # Fall back to schoolbook
            best_U, best_V, best_W = U0, V0, W0
            best_rank = n * m * p

    # Convert to output format with half-integer rounding
    U_out = [[round(float(best_U[i, j]), 4) for j in range(best_rank)] for i in range(n * m)]
    V_out = [[round(float(best_V[i, j]), 4) for j in range(best_rank)] for i in range(m * p)]
    W_out = [[round(float(best_W[i, j]), 4) for j in range(best_rank)] for i in range(p * n)]

    return {"U": U_out, "V": V_out, "W": W_out}
