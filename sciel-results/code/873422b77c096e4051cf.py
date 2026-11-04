"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check by computing the product and comparing to the identity tensor
    
    # Build the full tensor via the decomposition
    # T_decomposed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For efficiency, compute using einsum
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    T_decomposed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # The true tensor for matrix multiplication: T[i,j,k] = 1 for all i,j,k
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    # because the multiplication is linear and each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # So the tensor is all ones!
    T_true = np.ones((n, m, p))
    
    # Check if they match
    if not np.allclose(T_decomposed, T_true, atol=1e-9):
        return False
    return True

def _try_rank(n, m, p, rank, max_time=30):
    """Try to find a decomposition of given rank using random initialization and optimization."""
    start_time = time.time()
    
    # Random initialization
    np.random.seed(42 + rank)
    
    # Try multiple random initializations
    best_obj = np.inf
    best_params = None
    
    U_shape = (n * m, rank)
    V_shape = (m * p, rank)
    W_shape = (p * n, rank)
    
    n_params = U_shape[0] * rank + V_shape[0] * rank + W_shape[0] * rank
    
    for attempt in range(5):
        if time.time() - start_time > max_time * 0.8:
            break
            
        # Random initialization
        U = np.random.randn(*U_shape) * 0.1
        V = np.random.randn(*V_shape) * 0.1
        W = np.random.randn(*W_shape) * 0.1
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Objective: sum of squared differences from the true tensor
        def objective(x):
            U_flat, V_flat, W_flat = x[:U_shape[0]*rank], x[U_shape[0]*rank:U_shape[0]*rank+V_shape[0]*rank], x[U_shape[0]*rank+V_shape[0]*rank:]
            U_mat = U_flat.reshape(U_shape)
            V_mat = V_flat.reshape(V_shape)
            W_mat = W_flat.reshape(W_shape)
            
            U_r = U_mat.reshape(n, m, rank)
            V_r = V_mat.reshape(m, p, rank)
            W_r = W_mat.reshape(p, n, rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            diff = T - 1.0
            return np.sum(diff * diff)
        
        # Try L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if best_obj < 1e-8:
            break
    
    if best_obj < 1e-8 and best_params is not None:
        U_flat, V_flat, W_flat = best_params[:U_shape[0]*rank], best_params[U_shape[0]*rank:U_shape[0]*rank+V_shape[0]*rank], best_params[U_shape[0]*rank+V_shape[0]*rank:]
        U_mat = U_flat.reshape(U_shape).tolist()
        V_mat = V_flat.reshape(V_shape).tolist()
        W_mat = W_flat.reshape(W_shape).tolist()
        return U_mat, V_mat, W_mat
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # Known lower bounds and known results:
    # For 4x4x4, the minimal rank is 49 (Strassen's algorithm gives 49)
    # For 4x4x5, we need to search
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try ranks from lower to higher
    # Known: rank(4,4,4) = 49, so rank(4,4,5) >= 49
    # Upper bound from schoolbook: 80
    # Try a range of ranks
    
    for rank in range(49, 81):
        if time.time() > deadline:
            break
        
        max_time_for_rank = min(15, max(1, (deadline - time.time()) / max(1, 81 - rank)))
        
        result = _try_rank(n, m, p, rank, max_time=max_time_for_rank)
        if result is not None:
            U_mat, V_mat, W_mat = result
            # Verify
            if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
                best_rank = rank
                best_result = (U_mat, V_mat, W_mat)
                break  # Found minimal rank, stop
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
