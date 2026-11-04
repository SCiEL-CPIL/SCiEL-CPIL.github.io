"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (index into A)
    # b = j*p + k (index into B)  
    # c = k*n + i (index into C, using trace convention W[k*n+i])
    # T[a,b,c] = 1 if the indices are consistent, else 0
    # Consistency: a encodes (i,j), b encodes (j,k), c encodes (k,i)
    # So for a given (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        # Compute the reconstructed tensor
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Outer product
            recon += np.einsum('a,b,c->abc', u, v, w)
        return np.allclose(recon, T, atol=1e-10)
    
    # Start with trivial decomposition
    U = np.zeros((n*m, trivial_rank), dtype=np.float64)
    V = np.zeros((m*p, trivial_rank), dtype=np.float64)
    W = np.zeros((p*n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    best_rank = trivial_rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Strategy: Try to find a lower rank decomposition
    # Use the fact that matrix multiplication can be done with fewer than n*m*p multiplications
    # For small cases, we can try systematic searches
    
    # Approach: Try ranks from trivial_rank-1 down to some minimum
    # For each rank, try to find a valid decomposition using iterative methods
    
    # A more practical approach for this specific case:
    # Try to use known algebraic structures
    
    # For M_{3,4,7}, let's try to see if we can achieve rank 83, 82, etc.
    # The key is that some of the "1" entries in T might be achievable with shared rank-1 terms
    
    # Let's try a randomized approach with structured initialization
    np.random.seed(42)
    
    # Try a few candidate ranks
    for target_rank in range(trivial_rank - 1, max(40, trivial_rank - 15), -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U_cand = np.random.randn(n*m, target_rank) * 0.1
            V_cand = np.random.randn(m*p, target_rank) * 0.1
            W_cand = np.random.randn(p*n, target_rank) * 0.1
            
            # Try to refine using gradient descent on the loss
            # Loss = ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
            
            # This is a non-convex problem, so we'll use a simple iterative approach
            # Fix one factor at a time and solve for the others
            
            # Actually, for exact integer/half-integer solutions, random real-valued
            # approaches won't work well. Let's try a different strategy.
            
            # Strategy: Start from trivial and try to merge columns
            # If two columns in the trivial decomposition can be combined
            # (i.e., their sum or difference gives a valid contribution), we can reduce rank
            
            pass
        
        # Alternative: Try to construct a specific low-rank decomposition
        # using algebraic identities
        
        # For now, if we haven't found a better one, continue
    
    # Fallback: Use the trivial decomposition
    # The trivial decomposition is always valid
    
    # Convert to lists for output
    U_out = best_U[:, :best_rank].tolist()
    V_out = best_V[:, :best_rank].tolist()
    W_out = best_W[:, :best_rank].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
