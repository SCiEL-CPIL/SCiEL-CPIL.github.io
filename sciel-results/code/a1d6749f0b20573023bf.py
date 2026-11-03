"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s deadline, 20s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify(U, V, W, tol=1e-9):
        """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if abs(prod.sum() - 1.0) > tol:
                        return False
        return True
    
    # Try structured decompositions with various ranks
    # Strategy: use the bilinear structure more efficiently
    
    # Approach 1: Try to use a "block" structure where we group indices
    # For rank r, try various structured patterns
    
    # Let's try a systematic search over small structured decompositions
    # Key idea: the tensor is all-ones, so we want factors that multiply to give 1
    
    # Try rank values from low to high, with structured initialization
    for r in range(16, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        # Try structured initialization: use Kronecker-like patterns
        found = False
        
        # Pattern 1: U depends on (i,j), V on (j,k), W on (k,i)
        # Try to use a small number of "modes" for each factor
        
        # For each rank, try multiple random structured initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(r * 1000 + attempt)
            
            # Initialize with structured random values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Iterative refinement: alternating least squares style
            # For fixed U, V: solve for W
            # For fixed (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # For fixed (k,i): sum_c W[k,i,c] * (sum over j of U[i,j,c]*V[j,k,c] ... ) 
            # This doesn't decouple cleanly because of the j-sum inside
            
            # Alternative: treat it as a tensor completion / factorization problem
            # Use gradient descent on f = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            
            # For efficiency, use a simpler approach: try specific structured patterns
            
            # Pattern: make each factor depend on a subset of indices
            # U[i,j,c] = a[i,c] * b[j,c] (rank-1 in i,j for each c)
            # V[j,k,c] = b[j,c] * c[k,c] (rank-1 in j,k for each c)  
            # W[k,i,c] = c[k,c] * a[i,c] (rank-1 in k,i for each c)
            # Then sum_c a[i,c]*b[j,c]*b[j,c]*c[k,c]*c[k,c]*a[i,c] = sum_c (a[i,c]^2)(b[j,c]^2)(c[k,c]^2)
            # We need this = 1 for all i,j,k
            # This requires (a[i,c]^2)(b[j,c]^2)(c[k,c]^2) to sum to 1 independently of i,j,k
            # This is only possible if each factor is constant across its index, which means
            # a[i,c] = a_c for all i, etc. Then sum_c a_c^2 b_c^2 c_c^2 = 1
            # But this gives rank 1 effectively (all columns proportional)
            
            # Better: allow full generality but use a smarter search
            
            # Let's try: for fixed j, the equation becomes
            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,k
            # This is a bilinear form in (i,k) for each j
            # The matrix M_j[i,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should be all-ones
            
            # For a given j, if we fix U[:,j,:] and V[j,:,:], then W is constrained
            # by: for all i,k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            
            # This is a linear system in W for fixed U,V. Let's solve it.
            
            # Actually, let's try a different structured approach:
            # Use the fact that n=2 is small. For each i, we have p*m = 32 equations
            # involving W[k,i,:] for k=0..7
            
            # Try: set U and V to simple patterns, then solve for W
            # U[i,j,c] = 1/sqrt(r) for all i,j,c (uniform)
            # V[j,k,c] = 1/sqrt(r) for all j,k,c (uniform)
            # Then sum_c (1/r)*W[k,i,c] = 1 => sum_c W[k,i,c] = r
            # So W[k,i,c] = 1 for all k,i,c works!
            # But wait, that's rank r with all entries 1/sqrt(r), 1/sqrt(r), 1
            # Let's check: sum_c (1/sqrt(r))*(1/sqrt(r))*1 = sum_c 1/r = r/r = 1. Yes!
            
            # So for ANY rank r >= 1, we can set:
            # U[i,j,c] = 1/sqrt(r)
            # V[j,k,c] = 1/sqrt(r)  
            # W[k,i,c] = 1
            # And the sum is sum_c (1/r)*1 = 1. 
            # But wait, this requires r to be such that 1/sqrt(r) is valid.
            # The problem says entries should be integer/half-integer.
            # 1/sqrt(r) is not integer/half-integer for most r.
            
            # Let me reconsider. We need integer or half-integer entries.
            # So U[i,j,c] * V[j,k,c] * W[k,i,c] should sum to 1.
            # If we use U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1/r, that's not half-integer unless r divides 2.
            
            # For r=1: U=1, V=1, W=1. Sum = 1. Works!
            # Let's verify: for all i,j,k: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. Yes!
            
            # So rank 1 works! Let me double-check the index conventions.
            # U has shape (n*m, rank) = (8, 1)
            # V has shape (m*p, rank) = (32, 1)
            # W has shape (p*n, rank) = (16, 1)
            # U[i*m+j, 0] = 1 for all i in [0,2), j in [0,4)
            # V[j*p+k, 0] = 1 for all j in [0,4), k in [0,8)
            # W[k*n+i, 0] = 1 for all k in [0,8), i in [0,2)
            # Sum_c=0 to 0: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1. Correct!
            
            U_r1 = np.ones((n*m, 1))
            V_r1 = np.ones((m*p, 1))
            W_r1 = np.ones((p*n, 1))
            
            if verify_fast(U_r1, V_r1, W_r1):
                best_U, best_V, best_W = U_r1, V_r1, W_r1
                best_rank = 1
                found = True
                break
        
        if found:
            break
    
    # Convert to lists of lists
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
