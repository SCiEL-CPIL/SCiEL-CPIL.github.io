"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    Tries to find a decomposition with rank < 18 using numerical optimization.
    Falls back to schoolbook rank-18 if no lower rank is found.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Build the target tensor T[i,j,k] where (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # T is indexed as: U-index = i*m+j, V-index = j*p+k, W-index = k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Try ranks from 16 up to 17
    for rank in [16, 17]:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_decomposition(T, rank, dim_u, dim_v, dim_w, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook rank 18
    rank = 18
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank_decomposition(T, rank, dim_u, dim_v, dim_w, deadline):
    """Try to find a rank-r decomposition of tensor T using numerical optimization."""
    
    max_iter = 8000
    base_lr = 0.02
    
    for restart in range(8):
        if time.time() > deadline:
            return None
        
        np.random.seed(12345 + restart * 7919)
        
        # Initialize random rank-1 components
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Normalize each rank-1 component
        for r in range(rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                s = (nu * nv * nw) ** (-1.0/3.0)
                U[:, r] *= s
                V[:, r] *= s
                W[:, r] *= s
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute current tensor approximation
            approx = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            
            # Compute residual
            residual = approx - T
            res_norm = np.linalg.norm(residual)
            
            if res_norm < 1e-10:
                return _format_result(U, V, W, rank)
            
            # Compute gradients
            grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W, optimize=True)
            grad_V = np.einsum('ijk,ir,kr->jr', residual, U, W, optimize=True)
            grad_W = np.einsum('ijk,ir,jr->kr', residual, U, V, optimize=True)
            
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm < 1e-14:
                break
            
            # Adaptive learning rate with decay
            lr = base_lr / (1.0 + it / 2000.0)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Final check
        approx = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        if np.linalg.norm(approx - T) < 1e-8:
            return _format_result(U, V, W, rank)
    
    return None


def _format_result(U, V, W, rank):
    """Convert numpy arrays to lists of lists."""
    U_list = [[float(U[i, r]) for r in range(rank)] for i in range(U.shape[0])]
    V_list = [[float(V[i, r]) for r in range(rank)] for i in range(V.shape[0])]
    W_list = [[float(W[i, r]) for r in range(rank)] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
