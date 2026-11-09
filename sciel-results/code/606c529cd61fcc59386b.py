"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _schoolbook(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that U⊗V⊗W decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = delta(i,j,k) in flattened form
    # T is indexed by (i from A, j from B) -> k output? 
    # Actually the tensor M[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the trace convention W[k*n+i, r])
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] should be 1 (this is the matrix mult tensor)
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, num_trials=50, seed=42):
    """Try to find a rank-`rank` decomposition via random init + local search."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_U = best_V = best_W = None
    
    for trial in range(num_trials):
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Compute error: for each (i,j,k), |sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1|
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    errors.append(val - 1.0)
        err = np.max(np.abs(errors))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            return best_U, best_V, best_W, err
    return best_U, best_V, best_W, best_err

def _try_structured_decomposition(n, m, p, rank):
    """
    Try a structured approach: use the fact that matrix multiplication can be 
    decomposed via Kronecker products or block structures.
    
    For n x m * m x p, one approach is to use the identity:
    If we can factor the indices cleverly, we might reduce rank.
    
    Here we try a simple structured ansatz:
    - Group the (i,j) pairs and (j,k) pairs
    - Use a small number of "intermediate" dimensions
    """
    # Try rank = n*p + m*p + n*m - something
    # Actually, let's try a different structured approach:
    # Use the fact that C = A @ B can be written as sum over j of A[:,j] * B[j,:]
    # This is already the schoolbook with rank = n*m*p
    
    # Try: can we share computation across different j values?
    # For a fixed i and k, C[i,k] = sum_j A[i,j]*B[j,k]
    # If we could compute partial sums more efficiently...
    
    # Let's try a rank that's a multiple of some structure
    # For now, return None to indicate no structured solution found
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound considerations:
    # - Rank must be at least max(n*p, m*p, n*m) in some sense
    # - For 3x5*5x7: n*p=21, m*p=35, n*m=15
    # - The tensor rank is at least the "slice rank" which is >= max(n*m, m*p, n*p) = 35
    # - But actual rank could be higher
    
    # Try various ranks from a reasonable lower bound up to schoolbook
    # Be conservative with time
    max_trials_per_rank = 20
    
    # Try ranks: start from a few above the lower bound
    # Lower bound is at least max(n*m, m*p, n*p) = 35
    # Let's try from 35 to schoolbook_rank - 1
    # But be careful with time
    
    ranks_to_try = []
    # Try some strategic ranks
    for r in range(35, min(schoolbook_rank, 80)):
        ranks_to_try.append(r)
    
    # Sort by rank (try smaller first)
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        U, V, W, err = _try_random_decomposition(n, m, p, rank, num_trials=max_trials_per_rank)
        
        if err < 1e-8:
            # Found a valid decomposition!
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # Try to go lower
            continue
        
        # If we found something better than schoolbook but not exact, keep searching
        # but don't use it unless it's exact
    
    # Final verification
    if best_rank < schoolbook_rank:
        if _verify(best_U, best_V, best_W, n, m, p):
            # Convert to lists for output
            U_out = best_U.tolist()
            V_out = best_V.tolist()
            W_out = best_W.tolist()
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    U_out = U_sb.tolist()
    V_out = V_sb.tolist()
    W_out = W_sb.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
