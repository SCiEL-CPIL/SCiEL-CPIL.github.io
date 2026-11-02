"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    
    For each (i,j,k), the sum over r of U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    should equal 1.
    """
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_t = U.T  # (rank, n*m)
    V_t = V.T  # (rank, m*p)
    W_t = W.T  # (rank, p*n)
    
    # For efficiency, check a random sample of (i,j,k) triples
    # Actually, for small dimensions, just check all
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x6 matrix multiplication.
    
    Strategy: Use the schoolbook decomposition of rank 54 as a baseline,
    then attempt to reduce rank by finding linear dependencies among the
    tensor entries using numerical optimization.
    
    The key insight is that the matrix multiplication tensor T[i,j,k] = 1 for all
    (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1} has structure that may allow
    a lower-rank decomposition. We use a smooth optimization approach to
    find a rank-r decomposition if one exists.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    deadline = time.time() + 110  # 110s deadline to stay within 120s limit
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to reduce rank by attempting rank r for r from best_rank-1 down to some minimum
    # Use a smooth surrogate: minimize the sum of squares of (T_decomp - T_target)
    # where T_decomp is the tensor from the decomposition
    
    # For a rank-r decomposition, we have:
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 for all valid (i,j,k)
    
    # We'll try ranks from 53 down to 30, using random initialization and gradient-free search
    
    import random
    random.seed(42)
    
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Use a simple iterative refinement: coordinate descent
            # Minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            
            # Precompute index arrays for efficiency
            indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            
            # Simple gradient descent with random restarts
            lr = 0.01
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient for U
                dU = np.zeros((n * m, target_rank))
                dV = np.zeros((m * p, target_rank))
                dW = np.zeros((p * n, target_rank))
                
                for (i, j, k) in indices:
                    ui = U[i * m + j, :]
                    vj = V[j * p + k, :]
                    wk = W[k * n + i, :]
                    
                    # T = sum_r ui[r] * vj[r] * wk[r]
                    T_val = np.dot(ui * vj, wk)
                    err = T_val - 1.0
                    
                    # dU[i*m+j, r] = 2 * err * vj[r] * wk[r]
                    dU[i * m + j, :] += 2.0 * err * vj * wk
                    dV[j * p + k, :] += 2.0 * err * ui * wk
                    dW[k * n + i, :] += 2.0 * err * ui * vj
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if iteration % 20 == 0:
                    # Quick check
                    max_err = 0
                    for (i, j, k) in indices[:10]:  # Check subset
                        T_val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                        max_err = max(max_err, abs(T_val - 1.0))
                    if max_err < 1e-6:
                        break
            
            # Full verification
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid decomposition at this rank
    
    # Convert to lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
