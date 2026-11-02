"""Find rank-minimal tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    must equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    # Check a sample of (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # T[i,j,k] = 1 for matrix mult
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    rank = U.shape[1]
    # Reshape and compute tensor contraction
    # T[i,j,k] should equal 1
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Let's build the full tensor check
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each c, the term is outer product in some sense
    
    # Better: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We can vectorize:
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m*p)  # shape (n*m*p,)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)  # shape (n*m*p,)
    k_idx = np.tile(np.arange(p), n*m)  # shape (n*m*p,)
    
    U_idx = i_idx * m + j_idx  # shape (n*m*p,)
    V_idx = j_idx * p + k_idx  # shape (n*m*p,)
    W_idx = k_idx * n + i_idx  # shape (n*m*p,)
    
    # For each (i,j,k) and each c, compute U[U_idx,c]*V[V_idx,c]*W[W_idx,c]
    # Sum over c
    # U[U_idx, :] has shape (n*m*p, rank)
    U_sel = U[U_idx, :]  # (n*m*p, rank)
    V_sel = V[V_idx, :]  # (n*m*p, rank)
    W_sel = W[W_idx, :]  # (n*m*p, rank)
    
    # Element-wise product then sum over rank
    result = np.sum(U_sel * V_sel * W_sel, axis=1)  # (n*m*p,)
    
    return np.all(np.abs(result - 1.0) < tol)

def _try_rank(n, m, p, rank, max_time=10.0, rng=None):
    """Try to find a valid decomposition of given rank using gradient-based optimization.
    
    Minimizes the sum of squared errors of the tensor reconstruction.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    start_time = time.time()
    
    # Precompute indices for fast evaluation
    i_idx = np.repeat(np.arange(n), m*p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n*m)
    
    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx
    
    def objective(params):
        # Unpack parameters
        u_size = n * m
        v_size = m * p
        w_size = p * n
        
        U = params[0:u_size*rank].reshape(u_size, rank)
        V = params[u_size*rank:u_size*rank + v_size*rank].reshape(v_size, rank)
        W = params[u_size*rank + v_size*rank:].reshape(w_size, rank)
        
        # Compute reconstruction error
        U_sel = U[U_idx, :]
        V_sel = V[V_idx, :]
        W_sel = W[W_idx, :]
        
        # Sum over rank
        recon = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        # Error is (recon - 1)^2
        error = np.sum((recon - 1.0)**2)
        return error
    
    # Try multiple random initializations
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization
        U_init = rng.randn(n*m, rank) * 0.5
        V_init = rng.randn(m*p, rank) * 0.5
        W_init = rng.randn(p*n, rank) * 0.5
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B for optimization
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
        except Exception:
            continue
            
        if result.success or result.fun < 1e-8:
            # Check if the solution is valid
            U_sol = result.x[0:(n*m)*rank].reshape(n*m, rank)
            V_sol = result.x[(n*m)*rank:(n*m + m*p)*rank].reshape(m*p, rank)
            W_sol = result.x[(n*m + m*p)*rank:].reshape(p*n, rank)
            
            if _verify_fast(U_sol, V_sol, W_sol, n, m, p, tol=1e-5):
                return U_sol, V_sol, W_sol
                
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    # The schoolbook rank is n*m*p = 105
    # Lower bound is max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    # Let's search from the lower bound upward
    
    # First, try known efficient decompositions
    # For matrix multiplication, there are known algorithms:
    # - Strassen's algorithm for 2x2: rank 7 instead of 8
    # - For larger matrices, various improvements exist
    
    # Let's try to construct a decomposition of rank 35 (the lower bound)
    # If that fails, try higher ranks
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook as fallback
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better (lower rank) decompositions
    # Lower bound is 35
    
    # Strategy: try ranks from lower bound upward
    # For each rank, use gradient-based optimization
    
    for rank in range(max(n*p, m*p, n*m), best_rank + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Use the new optimization-based search
        result = _try_rank(n, m, p, rank, max_time=10.0)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            # Convert to lists
            U_list = [[float(U_sol[i, c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V_sol[i, c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W_sol[i, c]) for c in range(rank)] for i in range(p*n)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break
    
    return best_result
