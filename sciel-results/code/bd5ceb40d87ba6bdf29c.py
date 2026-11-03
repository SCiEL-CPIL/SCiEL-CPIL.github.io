"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check dimensions
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if i,j,k form a valid multiplication term, 0 otherwise
    # Actually for matrix multiplication: (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] should be 1 for all valid (i,j,k) triples
    
    # Let's verify by checking that for random matrices, the product matches
    rng = np.random.RandomState(42)
    for _ in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_arr[:, r]
            v_vec = V_arr[:, r]
            w_vec = W_arr[:, r]
            
            # u_vec is indexed by (i,j) -> reshape to (n,m)
            # v_vec is indexed by (j,k) -> reshape to (m,p)  
            # w_vec is indexed by (k,i) -> reshape to (p,n)
            
            # For each r, we add outer product contributions
            # The contribution to C[i,k] from term r is:
            # sum_j U[i*m+j,r] * V[j*p+k,r] * ... 
            # Actually, the standard formulation:
            # C[i,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,k]*V[j*p+k,r]) * W[k*n+i,r]
            # No, that's not right either.
            
            # Standard tensor decomposition for matmul:
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # And T[i,j,k] = 1 for all i,j,k (since (AB)[i,k] = sum_j A[i,j]B[j,k])
            
            # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Direct check of the tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook (trivial) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _try_structured_decompositions(n, m, p, deadline):
    """Try various structured low-rank decompositions."""
    best = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Try ranks from lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can do better with structure
    
    for rank in range(21, min(40, deadline_check(n, m, p))):
        if time.time() > deadline - 30:
            break
            
        # Try to find a decomposition of this rank
        # Use a simple iterative approach: initialize randomly and optimize
        
        found = _find_decomposition_rank(n, m, p, rank, deadline)
        if found is not None:
            if rank < best_rank:
                best_rank = rank
                best = found
                # Can't get better than this for now, but continue to see if we can find even lower
                # Actually, we want the minimum, so once we find one at rank R, 
                # we should try to find one at rank R-1, R-2, etc.
                # But for now, let's just return the first one we find
                break
    
    if best is None:
        return _build_schoolbook(n, m, p)
    return best

def deadline_check(n, m, p):
    """Helper to check deadline."""
    return 150

def _find_decomposition_rank(n, m, p, rank, deadline):
    """Try to find a valid decomposition of given rank using optimization."""
    # This is a non-convex problem. We'll use a simple gradient-based approach
    # or try known structured forms.
    
    # For now, return None to indicate we couldn't find one quickly
    # In a more complete implementation, we would use:
    # 1. Random initialization
    # 2. Alternating least squares or gradient descent
    # 3. Check convergence
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, get the schoolbook solution as fallback
    schoolbook = _build_schoolbook(n, m, p)
    
    # Try to find a better structured decomposition
    if time.time() < deadline - 60:
        result = _try_structured_decompositions(n, m, p, deadline)
        if result is not None and len(result["U"][0]) < len(schoolbook["U"][0]):
            # Verify before returning
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Return schoolbook as guaranteed valid solution
    return schoolbook
# EVOLVE-BLOCK-END
