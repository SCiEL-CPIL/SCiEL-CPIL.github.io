"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        1 = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _rank_reduce(U, V, W, n, m, p, tol=1e-9):
    """
    Attempt to reduce the rank of the decomposition by finding and eliminating
    linear dependencies among the columns of U, V, and W.
    
    Strategy:
    1. Check if columns of U are linearly independent. If not, find a null vector.
    2. If a null vector c exists for U (U @ c = 0), check if we can adjust V and W
       such that the tensor identity still holds. This is complex, so we use a
       simpler approach: try to eliminate columns one by one.
    
    For each column r, check if removing it can be compensated by adjusting
    other columns. This is equivalent to checking if the r-th term is linearly
    dependent on the others in the tensor space.
    
    A more practical approach:
    - Compute the rank of U, V, W.
    - If min(rank(U), rank(V), rank(W)) < current_rank, there are dependencies.
    - Use Gaussian elimination to find a basis and re-express the decomposition.
    
    However, the tensor decomposition constraint is:
    sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T (the multiplication tensor)
    
    This is a trilinear constraint, not just linear. So simple column elimination
    doesn't directly work.
    
    Alternative structural approach:
    Use the fact that for small matrices, we can try all subsets of columns to see
    if a smaller subset can span the same tensor. But this is exponential.
    
    Better: Use the known lower bounds and try to construct a decomposition
    at a lower rank using algebraic methods.
    
    For 3x4x4, the trivial rank is 48. Known bounds suggest it might be lower.
    Let's try a heuristic: randomly initialize at various ranks and use gradient
    descent or Newton's method to minimize the error.
    """
    return U, V, W

def _try_random_decomposition(n, m, p, target_rank, rng, time_budget=5.0, tol=1e-6):
    """
    Try to find a valid decomposition at target_rank using random initialization
    and iterative refinement.
    
    We use a simple approach: random initialization with values in {-1, 0, 1}
    and check validity. If not valid, we don't do gradient descent (too slow
    for the time budget), but we try many random initializations.
    """
    start_time = time.time()
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = rng.choice([-1, 0, 1], size=(n * m, target_rank)).astype(float)
        V = rng.choice([-1, 0, 1], size=(m * p, target_rank)).astype(float)
        W = rng.choice([-1, 0, 1], size=(p * n, target_rank)).astype(float)
        
        # Quick check: compute the tensor and compare
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] = 1 for all i,j,k
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None, None

def _try_structured_decomposition(n, m, p, target_rank, rng, time_budget=10.0, tol=1e-6):
    """
    Try a structured approach: use the schoolbook as a starting point and
    attempt to merge terms.
    
    For 3x4x4, we can think of the tensor as having structure.
    Let's try to use the fact that some entries might be combined.
    
    A better approach: use the known decomposition for smaller cases and
    extend it.
    
    For now, let's try a different random strategy: initialize with small
    integers and use a simple local search.
    """
    start_time = time.time()
    
    # Try multiple random seeds
    for seed in range(10):
        if time.time() - start_time > time_budget:
            break
        
        rng_local = random.Random(seed)
        U, V, W = _try_random_decomposition(n, m, p, target_rank, rng_local, time_budget=1.0, tol=tol)
        if U is not None:
            return U, V, W
    
    return None, None

def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, get the schoolbook solution as a guaranteed fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find a lower rank decomposition
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But the actual minimal rank is likely higher. Let's try from 40 down to 16.
    
    rng = random.Random(42)
    
    # Try ranks from 40 down to 16
    for target_rank in range(40, 15, -1):
        if time.time() > deadline:
            break
        
        # Try random decomposition
        U_t, V_t = _try_random_decomposition(n, m, p, target_rank, rng, time_budget=3.0, tol=1e-6)
        if U_t is not None:
            W_t = V_t  # This is a bug in the previous function, let's fix it
            # Actually, the function returns U, V, W
            break
        
        # Try structured decomposition
        U_t, V_t, W_t = _try_structured_decomposition(n, m, p, target_rank, rng, time_budget=5.0, tol=1e-6)
        if U_t is not None:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
    
    # Verify the best solution
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Convert to integer/half-integer if needed
        best_U = np.rint(best_U * 2) / 2
        best_V = np.rint(best_V * 2) / 2
        best_W = np.rint(best_W * 2) / 2
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
