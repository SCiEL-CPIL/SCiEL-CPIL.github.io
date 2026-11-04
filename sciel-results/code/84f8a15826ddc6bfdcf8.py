"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor."""
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Construct the target multiplication tensor T[i, j, k] = 1 for all i,j,k
        # The decomposition should satisfy: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Reshape U: (n*m, rank) -> (n, m, rank)
        U_t = U_np.reshape(n, m, rank)
        # Reshape V: (m*p, rank) -> (m, p, rank)
        V_t = V_np.reshape(m, p, rank)
        # Reshape W: (p*n, rank) -> (p, n, rank)
        W_t = W_np.reshape(p, n, rank)
        
        # For each (i, j, k), compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a contraction over c
        # Let's do it efficiently with einsum or explicit loops for small sizes
        n, m, p = 2, 3, 7
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m + j, c] * V_np[j*p + k, c] * W_np[k*n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        errors.append((i, j, k, val))
                    if errors and len(errors) > 10:
                        return False
        return len(errors) == 0
    except Exception:
        return False

def _try_rank_search(n, m, p, target_rank, time_budget=15.0, n_restarts=5, seed=42):
    """
    Try to find a decomposition of rank target_rank using random search + local refinement.
    """
    if target_rank < max(n*m, m*p, p*n):
        # Absolute lower bounds
        target_rank = max(n*m, m*p, p*n)
    
    start_time = time.time()
    
    # Known lower bounds for matrix multiplication rank:
    # For 2x3x7, schoolbook is 42.
    # Strassen-like reductions might help.
    # Let's just search.
    
    U_list = []
    V_list = []
    W_list = []
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + restart)
        
        # Initialize with random values
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        # Try a few alternating least squares style updates
        # Minimize ||T - sum_c U[:,c] outer V[:,c] outer W[:,c]||
        # But T is the identity-like tensor for matrix mult.
        
        # Actually, let's use a more structured approach.
        # For matrix multiplication, the tensor has symmetries.
        # Let's try to find a set of rank-1 tensors that sum to the multiplication tensor.
        
        # Simple random perturbation and gradient descent-like approach
        # But implementing exact gradient for tensor decomposition is complex.
        # Instead, let's try structured initializations.
        
        # Try: initialize W with identity-like pattern
        # Actually, let's just do random search with verification.
        # This is unlikely to find exact solutions, but it's a baseline.
        
        # More productive: try known algebraic constructions
        # For 2x3x7, we can try to factor out common subexpressions.
        
        # Example: A(2x3) * B(3x7)
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # Let's look for a decomposition using rank-1 updates
        # The multiplication tensor T has T[i,j,k] = 1 if (i,j,k) corresponds to (A row, A col, B col, B row)
        # In our indexing: i (A row), j (A col/B row), k (B col)
        
        pass # Skip this placeholder, use the method below
    
    # Try a more systematic approach:
    # Start from schoolbook (rank 42) and try to merge terms
    # This is equivalent to finding a lower rank decomposition
    
    # For now, let's just try to construct one with a different structure
    # Maybe using block decomposition?
    
    if time.time() - start_time < time_budget - 1.0:
        # Try a simple structured approach:
        # Since n=2, m=3, p=7
        # Can we do better than 42?
        # Rank lower bound: max(2*3, 3*7, 7*2) = max(6, 21, 14) = 21
        # But known bounds are higher.
        
        # Let's try to find a rank-21 to rank-41 decomposition by search
        # This is hard, so let's be conservative and just return schoolbook
        pass
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    # Start with schoolbook decomposition (guaranteed valid, rank 42)
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    # Try to find a lower rank decomposition
    # Given the difficulty of exact rank minimization for matrix multiplication,
    # we'll attempt a limited search.
    
    start_time = time.time()
    
    # Try ranks from just below schoolbook down to theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # 21
    # But let's be realistic: try a few ranks
    
    # For 2x3x7, known best might be 42 or slightly less
    # Let's try to improve by 1-5 ranks if possible
    
    # Strategy: try to construct a decomposition of rank r for r in [best_rank-1, ...]
    # Using a structured random search
    
    for target_rank in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() - start_time > 120:  # Leave 30s margin
            break
        
        # Try a simple random search for this rank
        success = _find_decomposition_for_rank(n, m, p, target_rank, time_budget=20.0)
        if success is not None:
            U_found, V_found, W_found = success
            if _verify_decomposition(U_found, V_found, W_found, n, m, p):
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = target_rank
                # Found a better one, try to go lower
                continue
        else:
            # No decomposition found for this rank
            if time.time() - start_time > 100:
                break
            # Move to next lower rank
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _find_decomposition_for_rank(n, m, p, target_rank, time_budget=10.0):
    """
    Attempt to find a decomposition of the matrix multiplication tensor with the given rank.
    Uses random initialization and verification.
    Returns (U, V, W) as lists, or None if not found.
    """
    start_time = time.time()
    
    # Theoretical lower bound check
    if target_rank < max(n * m, m * p, p * n):
        return None
    
    # For matrix multiplication, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try several random initializations
    n_attempts = 10
    for attempt in range(n_attempts):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(123 + attempt)
        
        # Initialize with sparse random structure
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Assign random +/-1 values in sparse pattern
        # Each column should be a valid "partial permutation" in some sense
        
        for c in range(target_rank):
            # Pick random indices
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            
            # Sometimes add negative terms to allow cancellation
            if rng.random() < 0.3:
                U[i * m + j, c] = -1.0
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None
# EVOLVE-BLOCK-END
