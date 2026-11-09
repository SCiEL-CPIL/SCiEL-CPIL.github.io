"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k])
                # Need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # This is a triple product, not just U·V
                # Recompute properly
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_r(n, m, p, r, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using iterative optimization.
    
    We parameterize U, V, W as r-column matrices and minimize the residual:
    residual = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    Use Gauss-Newton style updates.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    best_residual = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        # Compute residual and gradients
        # For each (i,j,k), f(i,j,k) = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        # df/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]
        # df/dV[j*p+k,c] = U[i*m+j,c]*W[k*n+i,c]
        # df/dW[k*n+i,c] = U[i*m+j,c]*V[j*p+k,c]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        residual = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    diff = val - 1.0
                    residual += diff * diff
                    for c in range(r):
                        grad_U[i*m+j, c] += diff * V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] += diff * U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] += diff * U[i*m+j, c] * V[j*p+k, c]
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < 1e-12:
            break
        
        # Gradient descent step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.9
        
    return best_residual, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with trivial solution as fallback
    U_triv, V_triv, W_triv = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # For all-ones tensor with this index structure, let's try from 15 upward
    # Actually the rank of the all-ones tensor of shape a×b×c is 1 if all dimensions are 1
    # Otherwise it's more complex. Let's just search.
    
    for r in range(1, min(best_rank, 30)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            res, U, V, W = _try_rank_r(n, m, p, r, max_iters=200, seed=seed + 100)
            if res < 1e-10:
                # Verify with exact check
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # Convert to list of lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
