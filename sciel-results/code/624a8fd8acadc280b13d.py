"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for M_{2,4,6}.
    
    Strategy: 
    1. Start with the schoolbook decomposition of rank n*m*p = 2*4*6 = 48.
    2. Use a "laser method" / embedding approach: try to find a lower-rank
       decomposition by exploiting structure.
       
    However, for 2x4x6, the known upper bound from Strassen-like recursions
    or direct search is the key. Let's first implement a robust verifier and
    a search that tries to reduce rank.
    
    Actually, let's use the known fact that M_{n,m,p} has rank at most n*m*p,
    and for small dimensions we can do a random search or use known results.
    
    For now, let's implement a correct schoolbook (rank 48) and then try to
    improve it via a random linear algebra approach or known lower bounds.
    
    Wait, the task says "score is the rank" and "lower is better". The current
    program has rank 48? No, the current program has rank n*m*p = 2*4*6 = 48.
    But the score is 1.0000, which means seed/value = 1.0, so value = seed = 48?
    No, the seed program's rank is 48, and score is 1.0. So seed = 48.
    
    Let's verify: the current program constructs rank = n*m*p = 2*4*6 = 48.
    Score = seed/value = 48/48 = 1.0.
    
    So we need to find a decomposition with rank < 48 to get score > 1.0.
    
    Known bounds for M_{2,4,6}:
    - Lower bound: at least max(2*4, 4*6, 2*6) = max(8, 24, 12) = 24.
    - Upper bound: 48 (schoolbook).
    
    Can we do better than 48? Yes, likely. Let's try a random search or
    use the structure of matrix multiplication.
    
    Actually, let's try a simple approach: use the fact that M_{2,4,6} can be
    embedded in M_{4,4,6} or use Strassen-like block methods.
    
    For now, let's implement a verifier and a random search that tries to
    find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 120  # 120 seconds max
    
    # First, let's get the schoolbook decomposition as a baseline
    def schoolbook_rank():
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
    
    U_sb, V_sb, W_sb = schoolbook_rank()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Verify the schoolbook decomposition
    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where indices are: U indexed by (i,j), V by (j,k), W by (k,i)
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # U[i*m+j, r] = u_{i,j,r}
        # V[j*p+k, r] = v_{j,k,r}
        # W[k*n+i, r] = w_{k,i,r}
        # T[i,j,k] = sum_r u_{i,j,r} * v_{j,k,r} * w_{k,i,r}
        # For matrix multiplication, T[i,j,k] = 1 if we're multiplying (i,j) of A and (j,k) of B to get (i,k) of C
        # Actually, the tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all valid i,j,k
        # Wait, no. The tensor is defined such that (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U, V, W are the factor matrices.
        
        # Let's check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomp(U_sb, V_sb, W_sb):
        # This should never happen, but just in case
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Now try to find a lower-rank decomposition
    # Use a random search approach: try random low-rank factorizations
    
    # Actually, let's try a more systematic approach.
    # For M_{2,4,6}, we can try to use the fact that it's a "thin" matrix multiplication.
    
    # Let's try ranks from 24 up to 47 and see if we can find a valid decomposition
    # using random initialization and optimization.
    
    # But optimization might be slow. Let's try a simpler approach first.
    
    # Actually, let's just return the schoolbook for now and see if we can improve
    # with a quick random search.
    
    # Try a few random low-rank attempts
    np.random.seed(42)
    for rank in range(24, min(48, best_rank)):
        if time.time() > deadline - 10:
            break
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            if verify_decomp(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        if best_rank <= rank:
            break
    
    # Convert to lists for the return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
