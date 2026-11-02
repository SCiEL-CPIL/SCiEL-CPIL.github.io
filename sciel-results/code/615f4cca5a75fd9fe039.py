"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Try ranks from 28 to 56
    # Lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    for rank in range(28, 57):
        if time.time() > deadline - 20:
            break
            
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline - 15:
                break
                
            res = _search_rank_gradient(n, m, p, rank, deadline, restart)
            if res is not None:
                return res
        
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_rank_gradient(n, m, p, rank, deadline, seed):
    """Search for a rank-`rank` decomposition using gradient descent and rounding."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n, m, rank) * 0.1
    V = np.random.randn(m, p, rank) * 0.1
    W = np.random.randn(p, n, rank) * 0.1
    
    lr = 0.05
    momentum = 0.9
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    # We want T_ijk = sum_c U_ijc V_jkc W_kic = 1
    # L = 0.5 * sum (1 - T)^2
    
    for it in range(1000):
        if time.time() > deadline - 10:
            break
            
        # Compute T and Error
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        E = 1.0 - T
        
        # Check if we are close enough to round
        if np.max(np.abs(E)) < 1e-2:
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            T_rounded = np.einsum('ijc,jkc,kic->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_rounded, 1.0, atol=1e-9):
                return _to_lists(U_rounded, V_rounded, W_rounded, n, m, p, rank)
                
        # Compute Gradients
        grad_U = -np.einsum('ijk,jkc,kic->ijc', E, V, W)
        grad_V = -np.einsum('ijk,ijc,kic->jkc', E, U, W)
        grad_W = -np.einsum('ijk,ijc,jkc->kic', E, U, V)
        
        # Update with Momentum
        vU = momentum * vU + lr * grad_U
        vV = momentum * vV + lr * grad_V
        vW = momentum * vW + lr * grad_W
        
        U -= vU
        V -= vV
        W -= vW
        
        # Decay learning rate
        lr *= 0.995
        
        # Periodic rounding check
        if it % 5 == 0 and np.max(np.abs(E)) < 0.1:
             U_rounded = np.round(U * 2) / 2.0
             V_rounded = np.round(V * 2) / 2.0
             W_rounded = np.round(W * 2) / 2.0
             T_rounded = np.einsum('ijc,jkc,kic->ijk', U_rounded, V_rounded, W_rounded)
             if np.allclose(T_rounded, 1.0, atol=1e-9):
                 return _to_lists(U_rounded, V_rounded, W_rounded, n, m, p, rank)

    return None

def _to_lists(U, V, W, n, m, p, rank):
    """Convert numpy arrays to lists of lists."""
    U_flat = U.reshape(n*m, rank)
    V_flat = V.reshape(m*p, rank)
    W_flat = W.reshape(p*n, rank)
    
    U_l = [[float(U_flat[i, c]) for c in range(rank)] for i in range(n*m)]
    V_l = [[float(V_flat[i, c]) for c in range(rank)] for i in range(m*p)]
    W_l = [[float(W_flat[i, c]) for c in range(rank)] for i in range(p*n)]
    return {"U": U_l, "V": V_l, "W": W_l}
# EVOLVE-BLOCK-END
