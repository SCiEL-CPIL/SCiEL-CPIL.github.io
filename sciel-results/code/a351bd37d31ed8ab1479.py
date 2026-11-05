"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x5."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 5]

# Known rank bounds / constructions.
# For (4,4,5), the schoolbook rank is 4*4*5 = 80.
# Strassen-style block decomposition can reduce this.
# We implement a systematic search for low-rank decompositions using
# the "laser method" style or known algebraic identities, but given time constraints,
# we use a constructive approach based on known results.

# For multiplying (n x m) by (m x p), the tensor has rank at most n*m*p.
# Lower bounds: rank >= max(n*p, m*p, n*m) = max(20, 20, 16) = 20.
# For (4,4,4), Strassen gives rank 7 < 8 (schoolbook).
# For (4,4,5), we can try to extend Strassen ideas.

# A known approach: use the fact that matrix multiplication tensor for (n,m,p)
# can be decomposed using block matrices. If we can find a rank-r decomposition,
# we're done.

# Given the 150s time limit, we'll try:
# 1. Start with schoolbook (rank 80)
# 2. Try to find a lower rank via randomized numerical search + rounding
# 3. Verify exactness with integer/rational arithmetic

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # The multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # We want U, V, W such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k}
    # Wait, the convention is: output W[k*n+i, r]
    # The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # T[i, j, k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    # A rank-r decomposition: sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Schoolbook: rank = n*m*p = 80
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        # Build the tensor from U, V, W
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Compare with the true multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using numerical optimization
    # We'll use a simple approach: try various ranks and use random initialization
    # with gradient-free search or least-squares fitting
    
    # The multiplication tensor T has entries T[i,j,k] = 1
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
    
    # This is a 3-way tensor decomposition problem.
    # We can flatten and solve iteratively.
    
    def try_rank(r, max_iters=50, seed=42):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        if time.time() > deadline - 5:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Initialize with random matrices
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        # Build the target tensor
        # T[i,j,k] = 1
        # We'll use an iterative ALS (Alternating Least Squares) approach
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i*m+j), solve least squares
            for idx in range(n*m):
                i = idx // m
                j = idx % m
                # U[idx, c] appears in T[i,j,k] for all k
                # T[i,j,k] = sum_c U[idx,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed i,j: sum_c U[idx,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is a linear system: U[idx, :] * M = 1
                # where M[c,k] = V[j*p+k,c] * W[k*n+i,c]
                M = np.zeros((r, p))
                for k in range(p):
                    for c in range(r):
                        M[c, k] = V[j*p + k, c] * W[k*n + i, c]
                # Solve M.T @ U[idx,:] = 1 (p equations, r unknowns)
                # Use least squares
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[idx, :] = sol
                except:
                    pass
            
            # Update V: for each (j*p+k), solve least squares
            for idx in range(m*p):
                j = idx // p
                k = idx % p
                # V[idx, c] appears in T[i,j,k] for all i
                # T[i,j,k] = sum_c U[i*m+j,c] * V[idx,c] * W[k*n+i,c]
                # For fixed j,k: sum_c V[idx,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                M = np.zeros((r, n))
                for i in range(n):
                    for c in range(r):
                        M[c, i] = U[i*m + j, c] * W[k*n + i, c]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V[idx, :] = sol
                except:
                    pass
            
            # Update W: for each (k*n+i), solve least squares
            for idx in range(p*n):
                k = idx // n
                i = idx % n
                # W[idx, c] appears in T[i,j,k] for all j
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[idx,c]
                # For fixed i,k: sum_c W[idx,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                M = np.zeros((r, m))
                for j in range(m):
                    for c in range(r):
                        M[c, j] = U[i*m + j, c] * V[j*p + k, c]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W[idx, :] = sol
                except:
                    pass
        
        # Check if the solution is valid
        if verify(U, V, W, tol=1e-8):
            return U, V, W
        return None
    
    # Try ranks from low to high
    # Lower bound is max(n*p, m*p, n*m) = 20
    # But for this specific problem, known results may give better bounds
    # Let's try a range of ranks
    
    # For efficiency, try some key ranks
    ranks_to_try = list(range(20, min(40, best_rank)))
    # Shuffle a bit to explore different seeds
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        for seed in range(5):  # Try a few random seeds
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iters=30, seed=seed)
            if result is not None:
                U, V, W = result
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition for this rank
    
    # Convert to lists of integers/half-integers if possible
    # Round to nearest 0.5 and verify
    def round_to_half(mat):
        return np.round(mat * 2) / 2
    
    U_r = round_to_half(best_U)
    V_r = round_to_half(best_V)
    W_r = round_to_half(best_W)
    
    if verify(U_r, V_r, W_r, tol=1e-9):
        best_U, best_V, best_W = U_r, V_r, W_r
    
    # Convert numpy arrays to lists
    def to_list(mat):
        return mat.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
