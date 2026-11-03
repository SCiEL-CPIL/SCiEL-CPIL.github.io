"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 120.0
_SAFETY_MARGIN = 10.0

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_find_decomposition(rank_target, n, m, p, max_iterations=1000):
    """Try to find a decomposition of the given rank using vectorized gradient descent."""
    if rank_target < 1:
        return None
    
    U = np.random.randn(n*m, rank_target)
    V = np.random.randn(m*p, rank_target)
    W = np.random.randn(p*n, rank_target)
    
    best_error = float('inf')
    lr = 0.01
    
    for it in range(max_iterations):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
            
        # Compute T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        T = np.zeros((n, m, p))
        for c in range(rank_target):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # T_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
            T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-9:
            return (U, V, W)
        
        if error < best_error:
            best_error = error
            
        # Gradient Descent
        E = 2 * (T - 1.0)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(rank_target):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            
            # dU_c[i,j] = sum_k E[i,j,k] Vc[j,k] Wc[k,i]
            dU[:, c] = np.einsum('ijk,jk,ki->ij', E, Vc, Wc).flatten()
            
            # dV_c[j,k] = sum_i E[i,j,k] Uc[i,j] Wc[k,i]
            dV[:, c] = np.einsum('ijk,ij,ki->jk', E, Uc, Wc).flatten()
            
            # dW_c[k,i] = sum_j E[i,j,k] Uc[i,j] Vc[j,k]
            dW[:, c] = np.einsum('ijk,ij,jk->ki', E, Uc, Vc).flatten()
            
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Try ranks from low to high
    for rank in range(30, schoolbook_rank):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        result = _try_find_decomposition(rank, n, m, p, max_iterations=500)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            # Round to nearest 0.5 for cleaner output
            U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
