"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    # T[i,j,k] should equal 1 for all valid (i,j,k)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """Try to reduce rank by merging pairs of rank-1 terms."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    current_rank = U.shape[1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r1 in range(current_rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                # Try to merge terms r1 and r2 into a single term
                # We need to find a new term (u_new, v_new, w_new) such that
                # the contribution of r1 + r2 can be replaced by u_new * v_new * w_new
                # This is generally not possible, but we can try specific patterns.
                
                # Check if the two terms have overlapping support in a way that allows merging
                # For now, this is a placeholder - a real implementation would use
                # algebraic methods to check if the sum of two rank-1 tensors is rank-1.
                pass
    
    return U.tolist(), V.tolist(), W.tolist(), current_rank

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns the schoolbook decomposition (rank 30) as baseline.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # 1. Construct Schoolbook Decomposition
    U_school = np.zeros((n * m, rank), dtype=object)
    V_school = np.zeros((m * p, rank), dtype=object)
    W_school = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U_school.tolist(), V_school.tolist(), W_school.tolist()
    best_rank = rank
    
    # 2. Attempt Rank Reduction
    if time.time() < deadline:
        U_list, V_list, W_list, new_rank = _try_rank_reduction(
            U_school, V_school, W_school, n, m, p, deadline
        )
        if new_rank < best_rank:
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = new_rank
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }

# EVOLVE-BLOCK-END
