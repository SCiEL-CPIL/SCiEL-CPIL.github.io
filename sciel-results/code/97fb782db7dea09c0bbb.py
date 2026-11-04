import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,k,j] = delta (1 if this is the right term)
    # Actually: T[i,j,k] = 1 for all i,j,k in the matmul tensor
    # Decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Wait, the convention is W[k*n+i, r], so W index is k*n+i
    # U[i*m+j, r], V[j*p+k, r]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p = 84
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Verify schoolbook
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        pass
    else:
        raise ValueError("Schoolbook verification failed")
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find a lower-rank decomposition using structured approach
    # For now, return the verified schoolbook decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
