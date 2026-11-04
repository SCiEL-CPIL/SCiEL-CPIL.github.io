"""Matrix multiplication 3x3 by 3x5 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 5]

def construct():
    """Find a low-rank exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (9, rank), (15, rank), (15, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This represents the matrix multiplication tensor where
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} for the identity
    # Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 
    # representing the coefficient of A[i,j]*B[j,k] in C[i,k]
    
    # The tensor to decompose: T[i, j, k] = 1 for all i,j,k
    # We want to write T as sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # and the reconstruction is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For exact matrix multiplication, we need:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # A rank-1 decomposition term is u * v * w where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # The reconstruction: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For this to equal 1 for all i,j,k, we need a specific structure.
    # The schoolbook method uses rank n*m*p with U[i*m+j, i*m+j] = 1, etc.
    
    # Let's try to find a lower rank decomposition using known results.
    # For 3x3 * 3x5, the minimum rank is not known exactly, but we can try various approaches.
    
    # Strategy: Try to find a decomposition with rank < 45 (schoolbook)
    # Use random initialization and local search, or try known structured decompositions.
    
    # First, let's verify the schoolbook works and get its rank as baseline
    rank_schoolbook = n * m * p  # 45
    
    # Try to find better decompositions using various strategies
    
    # Strategy 1: Try rank values from low to high, check if a valid decomposition exists
    # Using linear programming or random search
    
    # For small ranks, we can try to solve the system of equations
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a nonlinear system, so we'll use iterative methods
    
    # Let's try a greedy/iterative approach:
    # 1. Initialize U, V, W randomly
    # 2. Fix two, solve for the third (linear least squares)
    # 3. Iterate until convergence
    # 4. Check if the result is exact (within tolerance) and has integer/half-integer entries
    
    def try_decomposition(rank, n_iter=100, seed=42):
        """Try to find a rank-r decomposition using iterative methods."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to unit norm columns
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            # Fix V and W, solve for U
            # For each (i,j), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is: U[i*m+j, :] * diag(W[i, :]) * V[:, j*p:(j+1)*p].T ... 
            # Actually, let's set up the linear system properly.
            
            # For fixed V, W, the equation for U[i*m+j, :] is:
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k in [0,p)
            # Let M[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
            # Then sum_r M[i,j,k,r] * U[i*m+j, r] = 1
            
            # This is a linear system: M_ij * U[i*m+j, :] = 1 (vector of ones of length p)
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build the p x rank matrix
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            M[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    # Solve M * U[i*m+j, :] = ones(p)
                    try:
                        U_new[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Build n x rank matrix
                    M = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        V_new[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Build m x rank matrix
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W_new[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_new[i * m + j, r] * V_new[j * p + k, r] * W_new[k * n + i, r] for r in range(rank))
                        error += (val - 1.0) ** 2
            error = np.sqrt(error / (n * m * p))
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
        
        # Check if we got a good solution
        if best_error < 1e-6:
            return best_U, best_V, best_W, best_error
        return None
    
    # Try various ranks starting from a reasonable lower bound
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually, rank >= n*m*p / min(n*m, m*p, p*n) is not a valid bound.
    # A better lower bound: rank >= max(n, m, p) for the trivial case, but we can do better.
    # For matrix multiplication, rank >= n*m*p / max(n*m, m*p, p*n) is not standard.
    # Let's just try ranks from 10 to 45.
    
    best_result = None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Try ranks from low to high
    for rank in range(10, min(rank_schoolbook + 1, 30)):
        if time.time() > deadline - 20:
            break
        
        result = try_decomposition(rank, n_iter=50, seed=42 + rank)
        if result is not None:
            U, V, W, err = result
            if err < best_error:
                best_error = err
                best_rank = rank
                best_result = (U, V, W)
                if rank < 20:  # Found a good decomposition
                    break
    
    # If we found a good decomposition, try to clean it up to get integer/half-integer entries
    if best_result is not None:
        U, V, W = best_result
        
        # Try to round to nearest half-integers and check if it still works
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded version is still exact
        error_rounded = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r] for r in range(best_rank))
                    error_rounded += (val - 1.0) ** 2
        error_rounded = np.sqrt(error_rounded / (n * m * p))
        
        if error_rounded < 1e-6:
            U, V, W = U_rounded, V_rounded, W_rounded
            rank_final = best_rank
        else:
            # Use the non-rounded version if it's close enough
            # But we need integer/half-integer entries, so let's try harder
            # For now, fall back to schoolbook if we can't get clean entries
            U, V, W = None, None, None
            rank_final = rank_schoolbook
    
    # If we couldn't find a good decomposition, use schoolbook
    if U is None:
        rank = rank_schoolbook
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        rank_final = rank
    else:
        rank_final = best_rank
    
    # Convert to lists for the required output format
    U_list = [[float(U[i, r]) for r in range(rank_final)] for i in range(n * m)]
    V_list = [[float(V[i, r]) for r in range(rank_final)] for i in range(m * p)]
    W_list = [[float(W[i, r]) for r in range(rank_final)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
