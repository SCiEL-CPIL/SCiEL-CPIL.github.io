import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    
    # Schoolbook decomposition baseline (rank = n*m*p = 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
        """Vectorized check for faster evaluation."""
        rank = U.shape[1]
        # Compute all triple products efficiently
        # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def als_refine(U, V, W, n, m, p, max_iters=500, tol=1e-12, time_limit=None):
        """ALS refinement: alternate solving for each factor."""
        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Update U: for each (i,j), solve min sum_k (target - sum_r U[i,j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            for i in range(n):
                for j in range(m):
                    # Build A matrix: A[k,r] = V[j*p+k,r] * W[k*n+i,r]
                    # b[k] = 1.0
                    # Solve min ||A @ U[i,j,:] - b||^2
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve least squares
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = sol
            
            # Update V: for each (j,k), solve min sum_i (target - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = sol
            
            # Update W: for each (k,i), solve min sum_j (target - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = sol
            
            # Check convergence
            if check_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
                break
        
        return U, V, W
    
    # Try to find lower rank decompositions
    # Lower bound considerations: rank must be at least max(n*p, m*p, n*m) / something
    # For matrix multiplication, known lower bounds are complex
    # Let's try ranks from a reasonable lower bound up to the schoolbook rank
    
    # A simple lower bound: the rank is at least the dimension of the output space
    # which is n*p = 21, but tensor rank can be higher
    # Let's try a range of ranks
    
    # Start from lower ranks and work up
    # For 3x4x7, let's try ranks from 21 (n*p) up to 84
    # But we should be smart about which ranks to try
    
    # Try specific candidate ranks
    candidate_ranks = []
    # Include n*p = 21, and some other reasonable values
    for r in range(21, 85):
        candidate_ranks.append(r)
    
    # Sort by rank (try lower ranks first)
    candidate_ranks.sort()
    
    np.random.seed(42)
    
    for r in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations for this rank
        num_trials = 10 if r < 40 else 5
        
        found = False
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS refinement
            U_ref, V_ref, W_ref = als_refine(U, V, W, n, m, p, 
                                              max_iters=300, tol=1e-12,
                                              time_limit=deadline - 5.0)
            
            # Check if we found an exact decomposition
            if check_decomposition_fast(U_ref, V_ref, W_ref, n, m, p, tol=1e-9):
                # Verify with tighter tolerance
                if check_decomposition(U_ref, V_ref, W_ref, n, m, p, tol=1e-10):
                    best_rank = r
                    best_U = U_ref
                    best_V = V_ref
                    best_W = W_ref
                    found = True
                    break
        
        if found:
            # Found a decomposition at this rank, stop searching for lower ranks
            break
    
    # Final verification
    if not check_decomposition_fast(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = rank_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
