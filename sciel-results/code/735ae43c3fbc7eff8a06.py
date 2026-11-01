import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    # sum_r U[a,r] * V[b,r] * W[c,r]
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r
    
    # Efficient computation: reshape and use einsum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _build_strassen_2x2x2():
    """Build Strassen decomposition for 2x2x2."""
    # This is for reference, not directly used
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank
    # Lower bound: max(n*m, m*p, p*n) = max(4, 22, 22) = 22
    # Try ranks from 22 to 43
    
    for rank in range(22, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            if _verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
