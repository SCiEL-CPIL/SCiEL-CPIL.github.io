"""Matrix multiplication 4x5x6 rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k, l] = delta_{j,k} where i in [n], j in [m], k in [m], l in [p]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k}
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But actually the rank is at least the dimension of the space, which is larger
    
    # Use random search with ALS refinement
    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the tensor."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each (i,j,k,l), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{j,k}
        # This is expensive, so sample or check all
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        expected = 1.0 if j == k else 0.0
                        # sum over c
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
        return max_err < 1e-9
    
    def round_to_half_int(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def als_refine(U, V, W, rank, max_iters=100, tol=1e-10):
        """Alternating least squares to refine the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each row (i*m+j), solve for U[i*m+j, :]
            # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,k}
            # This is a linear system for each (i,j)
            for i in range(n):
                for j in range(m):
                    # For fixed i,j, we need: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = delta_{j,k} for all k
                    # Let A[k, c] = V[j*p+k, c] * W[k*n+i, c] for k in [m], c in [rank]
                    # Then A @ U[i*m+j, :]^T should give [delta_{j,0}, ..., delta_{j,m-1}, ...]
                    # Actually for each k, the equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = (1 if k==j else 0)
                    A = np.zeros((m, rank))
                    b = np.zeros(m)
                    for k in range(m):
                        b[k] = 1.0 if k == j else 0.0
                        for c in range(rank):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve A @ x = b, where x = U[i*m+j, :]
                    try:
                        x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V: for each row (j*p+k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    # For fixed j,k, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,k} for all i
                    # Let A[i, c] = U[i*m+j, c] * W[k*n+i, c] for i in [n], c in [rank]
                    # Then A @ V[j*p+k, :]^T should give [delta_{j,k} if... wait
                    # The equation is: for each i, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,k}
                    # But delta_{j,k} is 1 only if j==k, and this must hold for ALL i
                    # So the RHS is delta_{j,k} * ones(n)
                    A = np.zeros((n, rank))
                    b = np.zeros(n)
                    val = 1.0 if j == k else 0.0
                    for i in range(n):
                        b[i] = val
                        for c in range(rank):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W: for each row (k*n+i), solve for W[k*n+i, :]
            for k in range(m):
                for i in range(n):
                    # For fixed k,i, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,k} for all j
                    # Let A[j, c] = U[i*m+j, c] * V[j*p+k, c] for j in [m], c in [rank]
                    # Then A @ W[k*n+i, :]^T should give [delta_{0,k}, delta_{1,k}, ..., delta_{m-1,k}]
                    A = np.zeros((m, rank))
                    b = np.zeros(m)
                    for j in range(m):
                        b[j] = 1.0 if j == k else 0.0
                        for c in range(rank):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            if iteration % 10 == 0:
                err = compute_error(U, V, W, rank)
                if err < tol:
                    break
        
        return U, V, W
    
    def compute_error(U, V, W, rank):
        """Compute the maximum error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        expected = 1.0 if j == k else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
        return max_err
    
    # Search over ranks from a reasonable lower bound to schoolbook
    # Theoretical lower bound: the tensor is in R^{(n*m) x (m*p) x (p*n)}
    # Rank is at least max(n*m, m*p, p*n) / something... actually at least the "border rank"
    # For now, try from 30 up to schoolbook
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks in a strategic order
    ranks_to_try = list(range(30, min(60, rank_schoolbook + 1))) + list(range(60, rank_schoolbook + 1, 5))
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Multiple random initializations
        n_trials = 20
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # ALS refinement
            U, V, W = als_refine(U, V, W, rank, max_iters=50)
            
            # Check error
            err = compute_error(U, V, W, rank)
            if err < 1e-9:
                # Try to round to half-integers and verify
                U_rounded = round_to_half_int(U)
                V_rounded = round_to_half_int(V)
                W_rounded = round_to_half_int(W)
                
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded, rank)
                if err_rounded < 1e-9:
                    # Found a valid half-integer decomposition!
                    best_rank = rank
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    break
                else:
                    # Use the unrounded version if it's exact
                    # But we need half-integers... let's check if near
                    if err < 1e-6:
                        # Try snapping more aggressively
                        U_snapped = np.round(U * 2) / 2.0
                        V_snapped = np.round(V * 2) / 2.0
                        W_snapped = np.round(W * 2) / 2.0
                        err_snapped = compute_error(U_snapped, V_snapped, W_snapped, rank)
                        if err_snapped < 1e-9:
                            best_rank = rank
                            best_U = U_snapped.tolist()
                            best_V = V_snapped.tolist()
                            best_W = W_snapped.tolist()
                            break
            
            if best_rank <= rank:
                break
        
        if best_rank <= rank:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
