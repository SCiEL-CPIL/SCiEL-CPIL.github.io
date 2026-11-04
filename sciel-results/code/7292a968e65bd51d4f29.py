"""
Strategy: Numerical Optimization (Gradient Descent) for Tensor Rank Minimization.

Instead of hardcoding algebraic decompositions like Strassen, we treat the factor
matrices U, V, W as variables. We initialize them with random values and use
gradient descent to minimize the error between the reconstructed multiplication
tensor and the identity tensor.

We employ a "round-and-check" heuristic:
1. Run gradient descent for a fixed number of steps or until time limit.
2. Round the continuous values to the nearest 0.5 (half-integer).
3. Verify if the rounded decomposition is exact.
4. If not exact, try a new random initialization.
5. If we find a valid decomposition with rank < current best, keep it.
6. Return the best valid decomposition found.

This approach allows us to explore the solution space dynamically without needing
a priori knowledge of specific algebraic identities for 4x5x6.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# The target tensor T[i,j,k] = 1 if we are multiplying A[i,j] * B[j,k] -> C[i,k]
# In the decomposition U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
# We want T[i,j,k] = 1 for all valid i,j,k in the multiplication context.
# Actually, the tensor for matrix multiplication C = AB is:
# C[i,k] = sum_j A[i,j] B[j,k]
# The tensor T_{ijk} such that T_{ijk} A_{ij} B_{jk} = C_{ik} is T_{ijk} = 1.
# Wait, the standard definition is:
# C_{ik} = \sum_j A_{ij} B_{jk}
# The tensor T is defined by T_{i,j,k} = 1.
# The decomposition is: T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
# Note the indices in W: W[k*n+i, r] corresponds to W_{k,i,r}.

def _build_target_tensor():
    """
    Creates the target multiplication tensor T of shape (n, m, p).
    T[i, j, k] = 1.0 for all i, j, k.
    """
    return np.ones((n, m, p), dtype=float)

def _compute_error(U, V, W):
    """
    Computes the squared error between the reconstructed tensor and the target.
    U: (n*m, rank) -> reshape (n, m, rank)
    V: (m*p, rank) -> reshape (m, p, rank)
    W: (p*n, rank) -> reshape (p, n, rank) -> Note: W[k*n+i, r] is W_{k,i,r}
    
    Reconstructed: R[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    Target: T[i, j, k] = 1.0
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    # W is stored as W[k*n + i, r]. So W3[k, i, r] = W[k*n+i, r].
    W3 = W.reshape(p, n, rank)
    
    # R[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    R = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T = np.ones((n, m, p), dtype=float)
    error = np.sum((R - T) ** 2)
    return error

def _compute_gradients(U, V, W):
    """
    Computes gradients of the squared error with respect to U, V, W.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    R = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T = np.ones((n, m, p), dtype=float)
    diff = R - T  # Shape (n, m, p)
    
    # dE/dU[i,j,r] = 2 * sum_{k} diff[i,j,k] * V[j,k,r] * W[k,i,r]
    # dE/dV[j,k,r] = 2 * sum_{i} diff[i,j,k] * U[i,j,r] * W[k,i,r]
    # dE/dW[k,i,r] = 2 * sum_{j} diff[i,j,k] * U[i,j,r] * V[j,k,r]
    
    dU3 = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V3, W3)
    dV3 = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U3, W3)
    dW3 = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U3, V3)
    
    return dU3.reshape(n*m, rank), dV3.reshape(m*p, rank), dW3.reshape(p*n, rank)

def _verify(U, V, W):
    """
    Checks if the decomposition is exact with integer/half-integer entries.
    Returns True if valid.
    """
    # Check if entries are close to half-integers
    U_half = np.round(np.array(U) * 2) / 2
    V_half = np.round(np.array(V) * 2) / 2
    W_half = np.round(np.array(W) * 2) / 2
    
    if not (np.allclose(U, U_half, atol=1e-6) and 
            np.allclose(V, V_half, atol=1e-6) and 
            np.allclose(W, W_half, atol=1e-6)):
        return False
        
    # Check exact decomposition
    error = _compute_error(U_half, V_half, W_half)
    return error < 1e-6

def _optimize_rank(rank, max_steps=2000, lr=0.05, seed=0):
    """
    Attempts to find a valid decomposition of the given rank using gradient descent.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(max_steps):
        error = _compute_error(U, V, W)
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-10:
            break
            
        dU, dV, dW = _compute_gradients(U, V, W)
        
        # Simple gradient descent with adaptive learning rate
        # If error increases, reduce LR
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Normalize to prevent divergence
        norm_U = np.linalg.norm(U)
        norm_V = np.linalg.norm(V)
        norm_W = np.linalg.norm(W)
        if norm_U > 100:
            U /= norm_U * 100
        if norm_V > 100:
            V /= norm_V * 100
        if norm_W > 100:
            W /= norm_W * 100
            
    # Try rounding the best found
    U_half = np.round(best_U * 2) / 2
    V_half = np.round(best_V * 2) / 2
    W_half = np.round(best_W * 2) / 2
    
    if _verify(U_half, V_half, W_half):
        return U_half, V_half, W_half
        
    return None

def _schoolbook():
    """Fallback schoolbook decomposition."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks lower than schoolbook
    # Known lower bounds: max(n*m, m*p, p*n) = max(20, 30, 24) = 30.
    # Strassen-like reductions might get us to ~40-50.
    # We'll try a few specific ranks.
    
    ranks_to_try = [35, 40, 45, 50, 55, 60]
    
    for rank in ranks_to_try:
        if rank >= best_rank:
            continue
            
        if time.time() > deadline - 20:
            break
            
        # Try multiple seeds for this rank
        for seed in range(10):
            if time.time() > deadline - 10:
                break
                
            result = _optimize_rank(rank, max_steps=1000, lr=0.05, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to list for output format
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break  # Found valid decomposition for this rank, try lower
                
    return best_result

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    print(f"Found rank: {rank}")
    print(f"Error: {_compute_error(U, V, W)}")
