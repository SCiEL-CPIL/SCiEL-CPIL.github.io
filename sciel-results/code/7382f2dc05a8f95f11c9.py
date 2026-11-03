"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple.
    Actually, the tensor for matmul C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid i,j,k.
    The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    Wait, the convention is:
    U has shape (n*m, rank), indexed by (i*m + j)
    V has shape (m*p, rank), indexed by (j*p + k)
    W has shape (p*n, rank), indexed by (k*n + i)
    
    The tensor entry T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    And T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    """
    # The tensor is just all ones for valid indices
    return np.ones((n, m, p))

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Verify: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, rank, max_time=10.0):
    """Try to find a decomposition of given rank using numerical methods.
    This is a heuristic approach using random initialization and gradient-like updates."""
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # This is a 3-way tensor decomposition problem.
    # Let's use a simple iterative approach.
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # We want to minimize the error:
    # E = sum_{i,j,k} (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1)^2
    
    # Use a simple gradient descent approach
    lr = 0.01
    for iteration in range(1000):
        if time.time() - start_time > max_time:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    err = val - 1.0
                    
                    # Gradient with respect to U[u_idx, r]
                    dU[u_idx, :] += 2 * err * V[v_idx, :] * W[w_idx, :]
                    # Gradient with respect to V[v_idx, r]
                    dV[v_idx, :] += 2 * err * U[u_idx, :] * W[w_idx, :]
                    # Gradient with respect to W[w_idx, r]
                    dW[w_idx, :] += 2 * err * U[u_idx, :] * V[v_idx, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we got close to the target
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-6:
        # Convert to list of lists with rounded values
        U_list = [[round(U[i, r], 6) for r in range(rank)] for i in range(n * m)]
        V_list = [[round(V[i, r], 6) for r in range(rank)] for i in range(m * p)]
        W_list = [[round(W[i, r], 6) for r in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But we can do better with clever decompositions.
    
    # Try ranks from lower bound upward
    min_rank = max(n * m, m * p, p * n)
    
    # For 2x3x10, let's try some specific ranks
    # The schoolbook is 60. Let's try to reduce it.
    
    # Try a few ranks
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank_decomposition(n, m, p, rank, max_time=10.0)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = rank
                break  # Found the minimum rank we can achieve with this method
    
    return best
# EVOLVE-BLOCK-END
