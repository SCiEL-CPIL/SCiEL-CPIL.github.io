"""Program for rank-minimal tensor decomposition of 4x4 x 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i, j, k] = delta (i.e., T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k])
    # The tensor decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # Actually, the multiplication tensor T has T[i, j, k] = 1 for the "diagonal" in the appropriate sense
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i, j, k] should satisfy: the rank-1 decomposition sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = T[i,j,k]
    # where T[i,j,k] = 1 (it's the identity tensor for the multiplication)
    
    # Let's build the full tensor and check
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Compute the decomposition
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Reshape: U_r[i, j, c] = U[i*m+j, c], V_r[j, k, c] = V[j*p+k, c], W_r[k, i, c] = W[k*n+i, c]
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    
    # Compute sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a tensor contraction
    result = np.zeros((n, m, p))
    for c in range(rank):
        result += U_r[:, :, c][:, :, np.newaxis] * V_r[:, :, c][np.newaxis, :, :] * W_r[:, :, c][np.newaxis, np.newaxis, :]
    
    diff = np.max(np.abs(result - T))
    return diff < tol

def _try_rank_decomposition(n, m, p, target_rank, max_iters=2000, lr=0.01, seed=42):
    """Try to find a rank-`target_rank` decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, target_rank) * 0.1
    V = rng.randn(m * p, target_rank) * 0.1
    W = rng.randn(p * n, target_rank) * 0.1
    
    T = np.ones((n, m, p))
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute current decomposition
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        result = np.zeros((n, m, p))
        for c in range(target_rank):
            result += U_r[:, :, c][:, :, np.newaxis] * V_r[:, :, c][np.newaxis, :, :] * W_r[:, :, c][np.newaxis, np.newaxis, :]
        
        loss = np.sum((result - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Compute gradients
        # dL/dU_r[i,j,c] = 2 * sum_{k} (result[i,j,k] - T[i,j,k]) * V_r[j,k,c] * W_r[k,i,c]
        # dL/dV_r[j,k,c] = 2 * sum_{i} (result[i,j,k] - T[i,j,k]) * U_r[i,j,c] * W_r[k,i,c]
        # dL/dW_r[k,i,c] = 2 * sum_{j} (result[i,j,k] - T[i,j,k]) * U_r[i,j,c] * V_r[j,k,c]
        
        error = result - T
        
        dU = np.zeros_like(U_r)
        dV = np.zeros_like(V_r)
        dW = np.zeros_like(W_r)
        
        for c in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dU[i, j, c] += 2 * error[i, j, k] * V_r[j, k, c] * W_r[k, i, c]
                        dV[j, k, c] += 2 * error[i, j, k] * U_r[i, j, c] * W_r[k, i, c]
                        dW[k, i, c] += 2 * error[i, j, k] * U_r[i, j, c] * V_r[j, k, c]
        
        U -= lr * dU.reshape(n * m, target_rank)
        V -= lr * dV.reshape(m * p, target_rank)
        W -= lr * dW.reshape(p, n, target_rank).T.reshape(p * n, target_rank)
    
    return best_U, best_V, best_W, best_loss

def _rational_reconstruct(U_float, V_float, W_float, max_denom=1000):
    """Try to reconstruct rational values from floating point."""
    def to_frac(x):
        for d in range(1, max_denom + 1):
            n_val = round(x * d)
            if abs(x * d - n_val) < 1e-6:
                return Fraction(n_val, d)
        return Fraction(round(x))
    
    U_r = [[to_frac(x) for x in row] for row in U_float]
    V_r = [[to_frac(x) for x in row] for row in V_float]
    W_r = [[to_frac(x) for x in row] for row in W_float]
    
    return U_r, V_r, W_r

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try decreasing ranks from a reasonable upper bound
    # Known: schoolbook is 112. Try to find lower.
    # For 4x4x7, let's try ranks from 50 down to 20
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, provide schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    for target_rank in range(80, 15, -5):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            U, V, W, loss = _try_rank_decomposition(n, m, p, target_rank, 
                                                      max_iters=500, lr=0.005, seed=seed)
            
            if loss < 1e-8:
                # Try rational reconstruction
                U_r, V_r, W_r = _rational_reconstruct(U, V, W, max_denom=200)
                
                # Verify the rational decomposition
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try even lower ranks
            continue
        else:
            # If we can't find a decomposition at this rank, lower ranks are unlikely
            # But let's try a few more
            break
    
    return best_result
# EVOLVE-BLOCK-END
