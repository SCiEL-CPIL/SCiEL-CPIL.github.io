"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition for n x m * m x p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that the tensor decomposition matches the matrix multiplication tensor exactly."""
    R = U.shape[1]
    # Reshape U to (n, m, R), V to (m, p, R), W to (n, p, R)
    # Note: W is flattened as W[k*n + i, r] in the prompt, which corresponds to W_{ik} if we consider C_{ik}.
    # The prompt says W[k*n+i, r]. So index is k*n + i.
    # This means W is stored in column-major order for (k, i) if we view it as (p, n)?
    # Let's check: W[k, i] in math is usually C_{ik}.
    # If W is (p, n), then W[k, i] is at index k*n + i.
    # So Wr = W.reshape(p, n, R) gives Wr[k, i, r].
    Ur = U.reshape(n, m, R)
    Vr = V.reshape(m, p, R)
    Wr = W.reshape(p, n, R)
    
    # The matrix multiplication tensor T has T[i, j, k] = 1 if i,j,k are valid for multiplication?
    # No, T is in R^{nm} x R^{mp} x R^{np}.
    # T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k' ?
    # Actually, the product is C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k where the indices match?
    # The condition is:
    # sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1 for all i,j,k.
    # AND
    # sum_r U_{ij,r} V_{j'k,r} W_{ik',r} = 0 for any mismatch.
    
    # We check all combinations of i, j, k for A, and j', k for B, and i', k' for C?
    # Actually, we just need to check that the reconstructed tensor matches the target tensor.
    # Target tensor T_target[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j'*p + k
    # gamma = i'*n + k'
    
    # T_target is 1 if alpha=(i,j), beta=(j,k), gamma=(i,k) i.e., j matches and i matches and k matches.
    # Specifically, for a fixed r, the term is U_{ij,r} V_{j'k,r} W_{i'k',r}.
    # We sum over r.
    # The result should be 1 if (i,j) from A, (j,k) from B, (i,k) from C are consistent (j=j, i=i, k=k).
    # And 0 otherwise.
    
    # We can iterate over all i, j, k for the "valid" terms and ensure they are 1.
    # And iterate over "invalid" terms and ensure they are 0.
    
    # Valid terms:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                if abs(s - 1.0) > tol:
                    return False
                    
    # Invalid terms:
    # We need to check cases where indices don't match.
    # e.g., A_{ij}, B_{j'k}, C_{i'k'} with j != j' or i != i' or k != k'.
    # To be efficient, we can check a subset or all.
    # Total size: n*m * m*p * n*p = n^2 m^2 p^2.
    # For 3,3,4: 9 * 12 * 12 = 1296 terms. R=36. 1296*36 ~ 46k ops. Fast.
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            # Check if this is a valid multiplication term
                            # Valid if j == j2, i == i2, k == k2
                            if j == j2 and i == i2 and k == k2:
                                continue
                            s = 0.0
                            for r in range(R):
                                s += Ur[i, j, r] * Vr[j2, k, r] * Wr[k, i2, r]
                            if abs(s) > tol:
                                return False
    return True


def _try_random_rank(R, n, m, p, time_limit, seed_offset=0):
    """Attempt to find a rank-R decomposition via random initialization + gradient-free search.
    This is extremely unlikely to succeed for R < 36, but we try briefly."""
    if R >= n * m * p:
        return None
    start = time.time()
    rng = np.random.RandomState(42 + seed_offset)
    
    # Initialize with small random values
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Quick check
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # 1. Schoolbook Decomposition (Rank 36) - always valid
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p  # 36
    
    # 2. Try to find lower rank decompositions
    # Strategy: Try ranks from best_rank-1 downward, with decreasing time budgets
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return best_result
    
    # Try rank 35 with a short time budget
    time_budget_35 = min(10.0, remaining_time * 0.3)
    result_35 = _try_random_rank(35, n, m, p, time_budget_35, seed_offset=0)
    if result_35 is not None:
        U35, V35, W35 = result_35
        if _verify(U35, V35, W35, n, m, p):
            best_result = {
                "U": U35.tolist(),
                "V": V35.tolist(),
                "W": W35.tolist()
            }
            best_rank = 35
    
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return best_result
    
    # Try rank 34 with remaining time
    if best_rank > 34:
        time_budget_34 = min(20.0, remaining_time * 0.5)
        result_34 = _try_random_rank(34, n, m, p, time_budget_34, seed_offset=100)
        if result_34 is not None:
            U34, V34, W34 = result_34
            if _verify(U34, V34, W34, n, m, p):
                best_result = {
                    "U": U34.tolist(),
                    "V": V34.tolist(),
                    "W": W34.tolist()
                }
                best_rank = 34
    
    return best_result
# EVOLVE-BLOCK-END
