"""Tensor decomposition for 3x4x6 matrix multiplication."""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 6]

def _make_schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _snap_to_half_integer(arr, tol=1e-8):
    """Snap array values to nearest half-integer if close enough."""
    result = arr.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            val = result[i, j]
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < tol:
                result[i, j] = nearest
    return result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with the schoolbook decomposition as fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need to find rank-1 terms (u_c, v_c, w_c) such that
    # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all (i,j,k)
    
    # Strategy: Use a structured search over sparse rank-1 terms
    # Each rank-1 term is defined by its support patterns on U, V, W
    
    # Generate candidate sparse patterns
    # For U: subsets of {0,...,n*m-1} = {0,...,11}
    # For V: subsets of {0,...,m*p-1} = {0,...,23}
    # For W: subsets of {0,...,p*n-1} = {0,...,17}
    
    # A rank-1 term with support sets S_U, S_V, S_W contributes
    # to T[i,j,k] if i*m+j in S_U, j*p+k in S_V, and k*n+i in S_W
    
    # We want to cover all (i,j,k) triples with a minimum number of such terms
    # This is a set cover problem, but we can use a greedy approach with verification
    
    # First, let's try to find a decomposition using a more systematic approach
    # based on known lower bounds and structured patterns
    
    # Known result: For m x n matrix multiplication, the minimum rank is at least
    # max(m, n)^2 / min(m, n) roughly, but exact values depend on specific dimensions
    
    # For 3x4x6, let's try ranks from a reasonable lower bound upward
    # Lower bound: rank >= max(n*m, m*p, p*n) / min(n, m, p) roughly
    # But a better lower bound comes from the fact that each rank-1 term can
    # contribute to at most one "slice" in a structured way
    
    # Let's use a different approach: direct search with structured initialization
    
    # Generate structured rank-1 terms based on modular arithmetic patterns
    def _generate_structured_terms(n, m, p, rank):
        """Generate structured rank-1 terms for a given rank."""
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            # Use a structured pattern based on c
            # Pattern: for each c, assign values based on modular relationships
            for i in range(n):
                for j in range(m):
                    # U[i*m + j, c]
                    U[i * m + j, c] = 1.0 if (i * m + j) % rank == c else 0.0
            
            for j in range(m):
                for k in range(p):
                    # V[j*p + k, c]
                    V[j * p + k, c] = 1.0 if (j * p + k) % rank == c else 0.0
            
            for k in range(p):
                for i in range(n):
                    # W[k*n + i, c]
                    W[k * n + i, c] = 1.0 if (k * n + i) % rank == c else 0.0
        
        return U, V, W
    
    # Try structured decompositions for various ranks
    for target_rank in range(40, min(75, n * m * p + 1)):
        if time.time() > deadline - 10:
            break
        
        U, V, W = _generate_structured_terms(n, m, p, target_rank)
        
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If structured approach didn't work, try a more flexible search
    # Use local search / hill climbing on the factor matrices
    
    if best_rank == n * m * p:
        # Try local search from a random initialization
        for target_rank in range(50, min(75, n * m * p + 1)):
            if time.time() > deadline - 15:
                break
            
            found = False
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(target_rank * 1000 + attempt * 7 + 42)
                
                # Initialize with small random values
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
                
                # Local search: minimize the maximum absolute error
                best_loss = float('inf')
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    # Compute the error tensor
                    errors = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                pred = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                errors.append(pred - 1.0)
                    
                    errors = np.array(errors)
                    loss = np.max(np.abs(errors))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    if best_loss < 1e-10:
                        found = True
                        break
                    
                    # Compute gradients for max error (subgradient)
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    # Find the entry with maximum absolute error
                    max_idx = np.argmax(np.abs(errors))
                    i = max_idx // (m * p)
                    j = (max_idx % (m * p)) // p
                    k = max_idx % p
                    
                    sign = np.sign(errors[max_idx])
                    
                    # Gradient contribution from this entry
                    grad_U[i * m + j, :] = sign * V[j * p + k, :] * W[k * n + i, :]
                    grad_V[j * p + k, :] = sign * U[i * m + j, :] * W[k * n + i, :]
                    grad_W[k * n + i, :] = sign * U[i * m + j, :] * V[j * p + k, :]
                    
                    # Update
                    lr = 0.01 / (1 + step * 0.01)
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                if found:
                    best_rank = target_rank
                    best_U, best_V, best_W = best_U, best_V, best_W
                    break
            
            if found:
                break
    
    # Try to snap to half-integers
    U_snapped = _snap_to_half_integer(best_U)
    V_snapped = _snap_to_half_integer(best_V)
    W_snapped = _snap_to_half_integer(best_W)
    
    if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-10):
        best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
