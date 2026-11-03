import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)  
    W: shape (p*n, rank) = (12, 48)
    
    For each (i,j,k) triple, one rank-1 term:
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    
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

def _verify_decomposition(U, V, W):
    """Verify exact decomposition of the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[i,k,j] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    # We want T to be all ones.
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute T[i, k, j] = sum_t U[i, j, t] * V[j, k, t] * W[k, i, t]
    # This is a contraction over t, and we need to align i, j, k correctly.
    # U has indices (i, j)
    # V has indices (j, k)
    # W has indices (k, i)
    
    # Sum over t:
    # term[t] = U[i,j,t] * V[j,k,t] * W[k,i,t]
    # T[i,k,j] = sum_t term[t]
    
    # We can do this with einsum:
    # 'ijt,jkt,kit -> ikj'
    T = np.einsum('ijt,jkt,kit->ikj', U_reshaped, V_reshaped, W_reshaped)
    
    expected = np.ones((n, p, m), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-9)

def _round_to_half_int(arr):
    """Round numpy array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_optimize_rank(R, deadline, seed=0):
    """
    Attempt to find a rank-R decomposition using vectorized gradient descent.
    """
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    lr = 0.05
    steps = 2000 # More steps possible due to speedup
    
    for step in range(steps):
        if time.time() > deadline:
            return None
            
        # Compute T[i, k, j] = sum_t U[i, j, t] V[j, k, t] W[k, i, t]
        # Using einsum: 'ijt,jkt,kit->ikj'
        T = np.einsum('ijt,jkt,kit->ikj', U_reshaped, V_reshaped, W_reshaped)
        
        # Error E = sum (T - 1)^2
        # Gradient of E w.r.t U[i, j, t]:
        # dE/dU[i,j,t] = 2 * sum_{k} (T[i,k,j] - 1) * V[j,k,t] * W[k,i,t]
        # Let Err[i,k,j] = T[i,k,j] - 1
        Err = T - 1.0
        
        # dU_reshaped[i,j,t] = 2 * sum_k Err[i,k,j] * V_reshaped[j,k,t] * W_reshaped[k,i,t]
        # Indices: i, j, t fixed. Sum over k.
        # Err is (i, k, j)
        # V is (j, k, t)
        # W is (k, i, t)
        # We want result (i, j, t)
        # dU = 2 * einsum('ikj,jkt,kit->ijt', Err, V_reshaped, W_reshaped)
        
        dU = 2.0 * np.einsum('ikj,jkt,kit->ijt', Err, V_reshaped, W_reshaped)
        
        # dV[j,k,t] = 2 * sum_i Err[i,k,j] * U[i,j,t] * W[k,i,t]
        # Result (j, k, t)
        # dV = 2 * einsum('ikj,ijt,kit->jkt', Err, U_reshaped, W_reshaped)
        
        dV = 2.0 * np.einsum('ikj,ijt,kit->jkt', Err, U_reshaped, W_reshaped)
        
        # dW[k,i,t] = 2 * sum_j Err[i,k,j] * U[i,j,t] * V[j,k,t]
        # Result (k, i, t)
        # dW = 2 * einsum('ikj,ijt,jkt->kit', Err, U_reshaped, V_reshaped)
        
        dW = 2.0 * np.einsum('ikj,ijt,jkt->kit', Err, U_reshaped, V_reshaped)
        
        # Update
        U_reshaped -= lr * dU
        V_reshaped -= lr * dV
        W_reshaped -= lr * dW
        
        # Check convergence or error
        # if np.max(np.abs(Err)) < 1e-5:
        #    break
        
    # Round to half-integers
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)
    
    # Convert to lists
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return None

def _try_find_lower_rank(deadline):
    """Attempt to find a rank lower than 48 using optimization."""
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    current_rank = len(schoolbook["U"][0])
    
    # Try ranks from 47 down to 19
    # We start from 47 and go down.
    # If we find a valid one, return it.
    for R in range(current_rank - 1, 18, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few seeds for this rank
        # Since the problem is hard, we might need more seeds or better init
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_optimize_rank(R, deadline, seed)
            if result is not None:
                return result
                
    return None

def construct():
    """Construct the tensor decomposition with deadline guard."""
    start_time = time.time()
    deadline = start_time + 110  # 120s limit, leave 10s margin for safety
    
    # Start with schoolbook (rank 48)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank
    if time.time() < deadline - 5:
        result = _try_find_lower_rank(deadline)
        if result is not None:
            U, V, W = result
            new_rank = len(U[0])
            if new_rank < best_rank and _verify_decomposition(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = new_rank
    
    return best
# EVOLVE-BLOCK-END
