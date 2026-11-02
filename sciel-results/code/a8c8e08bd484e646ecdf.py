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
    """Try to find a valid decomposition of given rank using random initialization + gradient-free refinement.
    
    Uses the fact that for matrix multiplication, we can try structured initializations.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization with small values
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Initialize with random values
        U[:] = rng.gauss(0, 0.5)
        V[:] = rng.gauss(0, 0.5)
        W[:] = rng.gauss(0, 0.5)
        
        # Try a few iterations of a simple optimization
        # We want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        
        # Use a simple iterative approach: for each (i,j,k), adjust the rank-1 terms
        # This is a tensor decomposition problem.
        
        # Alternative: use the structure of matrix multiplication
        # For matrix mult, we know that rank n*m*p works (schoolbook)
        # We want to find if lower rank works.
        
        # Let's try a different approach: use known results
        # For 3x5*5x7, the lower bound is max(n*p, m*p, n*m) = max(21, 35, 15) = 35
        # Upper bound is n*m*p = 105
        
        # Try to find a decomposition using structured patterns
        pass
    
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
    # For each rank, try random search + refinement
    
    rng = np.random.RandomState(12345)
    
    for rank in range(max(n*p, m*p, n*m), best_rank + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Try random initialization with structured patterns
        found = False
        
        # Method 1: Try to use the Kronecker product structure
        # For matrix multiplication, we can think of it as:
        # C = A * B where A is n x m, B is m x p
        # The tensor is T[i,j,k] = delta (i.e., 1 for all valid i,j,k)
        
        # Try a greedy/iterative approach
        # Initialize with zeros
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # For each rank component c, we need to assign values
        # such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is a system of n*m*p equations with 3*rank*dim variables
        # (where dim is the respective dimension)
        
        # Let's try a simple iterative method:
        # For each (i,j,k), pick a random c and set values to satisfy
        # This won't work directly because of the coupling
        
        # Alternative: use the fact that for matrix multiplication,
        # we can factor through the middle dimension
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all i,j,k
        
        # A rank-1 decomposition of the all-ones tensor would be:
        # T[i,j,k] = sum_c u[i,j,c] * v[j,k,c] * w[k,i,c]
        
        # Let's try random search with a simple update rule
        for attempt in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U[:] = rng.randn(n*m, rank) * 0.1
            V[:] = rng.randn(m*p, rank) * 0.1
            W[:] = rng.randn(p*n, rank) * 0.1
            
            # Check if it works
            if _verify_fast(U, V, W, n, m, p, tol=1e-4):
                # Found a valid decomposition!
                found = True
                break
        
        if found:
            # Convert to lists
            U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break
    
    return best_result
